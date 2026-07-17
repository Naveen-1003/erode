import React, { useState, useEffect, useRef } from 'react';
import { 
  StyleSheet, 
  View, 
  Text, 
  TouchableOpacity, 
  Alert, 
  ActivityIndicator, 
  StatusBar,
  ScrollView
} from 'react-native';
import { Camera, CameraView } from 'expo-camera';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { WS_URL } from '../../../services/api';
import { Play, Square, Flame, Clock, Activity, Zap } from 'lucide-react-native';
import { DrawerMenuButton } from '../../../components/drawer-menu-button';

const EXERCISES = ['Squat', 'Pushups', 'Jumping Jacks', 'Running', 'Walking'];

// Isolated from parent re-renders caused by live-metric state updates.
// animateShutter={false} suppresses the 1-second visual flash from takePictureAsync.
const CameraPreview = React.memo(function CameraPreview({
  innerRef,
}: {
  innerRef: React.RefObject<CameraView>;
}) {
  return <CameraView ref={innerRef} style={styles.camera} animateShutter={false} />;
});

export default function CameraScreen() {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [selectedActivity, setSelectedActivity] = useState('Squat');
  const [isRecording, setIsRecording] = useState(false);
  const [loading, setLoading] = useState(false);

  // Real-time workout metrics
  const [liveCalories, setLiveCalories] = useState(0.0);
  const [liveDuration, setLiveDuration] = useState(0.0);
  const [liveIntensity, setLiveIntensity] = useState('Low');
  const [liveSpeed, setLiveSpeed] = useState(0.0);
  const [poseDetected, setPoseDetected] = useState(true);

  const [errorState, setErrorState] = useState('');

  const cameraRef = useRef<CameraView>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const streamIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const stoppingRef = useRef(false);
  const capturingRef = useRef(false);
  const router = useRouter();

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();

    return () => {
      stopIntervals();
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const stopIntervals = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);
  };

  const startWorkout = async () => {
    stoppingRef.current = false;
    setLoading(true);
    setErrorState('');
    try {
      const token = await AsyncStorage.getItem('user_token');
      if (!token) {
        Alert.alert('Authentication Error', 'Session expired. Please log in again.');
        router.replace('/auth/login');
        return;
      }

      // 1. Initialize WebSocket Connection
      const ws = new WebSocket(`${WS_URL}/api/predict/live?token=${token}`);
      wsRef.current = ws;

      // Connection timeout — surface a clear error if server is unreachable
      const wsTimeout = setTimeout(() => {
        if (wsRef.current?.readyState !== WebSocket.OPEN) {
          wsRef.current?.close();
          setErrorState('Connection timed out. Make sure the backend server is running and reachable.');
          setLoading(false);
        }
      }, 6000);

      ws.onopen = () => {
        clearTimeout(wsTimeout);
        console.log('[Burn-Ex Live] WebSocket opened.');

        const startTime = Date.now();
        ws.send(JSON.stringify({
          event: 'start_workout',
          activity: selectedActivity,
          timestamp: startTime / 1000
        }));

        setIsRecording(true);
        setLoading(false);
        setLiveCalories(0.0);
        setLiveDuration(0.0);
        setLiveIntensity('Low');
        setLiveSpeed(0.0);
        setPoseDetected(true);
        capturingRef.current = false;

        // Local duration counter — increments every second regardless of frame success.
        timerRef.current = setInterval(() => {
          setLiveDuration(prev => prev + 1);
        }, 1000);

        // 2. Capture real camera frames at 1 FPS and stream to backend for ML inference.
        let elapsed = 0.0;
        streamIntervalRef.current = setInterval(async () => {
          elapsed += 1.0;
          if (capturingRef.current || !cameraRef.current || ws.readyState !== WebSocket.OPEN) return;

          capturingRef.current = true;
          try {
            const photo = await cameraRef.current.takePictureAsync({
              quality: 0.2,
              base64: true,
              skipProcessing: true,
              shutterSound: false,
            });

            if (photo.base64 && ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({
                event: 'frame_image',
                timestamp: (startTime / 1000) + elapsed,
                image: photo.base64,
              }));
            }
          } catch (e) {
            console.warn('[Burn-Ex Live] Frame capture error:', e);
          } finally {
            capturingRef.current = false;
          }
        }, 1000); // 1 FPS
      };

      ws.onmessage = (e) => {
        let data: any;
        try {
          data = JSON.parse(e.data);
        } catch {
          return;
        }

        if (data.event === 'live_update') {
          setPoseDetected(data.pose_detected !== false);
          setLiveCalories(data.calories ?? 0);
          setLiveIntensity(data.intensity ?? 'Low');
          setLiveSpeed(data.movement_score ?? 0);
        } else if (data.event === 'workout_saved') {
          stopIntervals();
          setIsRecording(false);
          Alert.alert('Workout Saved!', `Total Burn: ${(data.calories ?? 0).toFixed(1)} kcal`, [
            { text: 'View Details', onPress: () => router.push(`/workout/${data.id}`) }
          ]);
        } else if (data.event === 'workout_discarded') {
          stopIntervals();
          setIsRecording(false);
          Alert.alert('Session Too Short', 'No workout data was recorded. Make sure you are visible in the camera and performing your exercise before finishing.');
        }
      };

      ws.onerror = (e) => {
        console.error('[Burn-Ex Live] WebSocket error:', e);
        if (!stoppingRef.current) {
          setErrorState('Connection to the backend lost. Please try again.');
          setLoading(false);
          setIsRecording(false);
          stopIntervals();
        }
      };

      ws.onclose = () => {
        console.log('[Burn-Ex Live] WebSocket closed.');
        setIsRecording(false);
        stopIntervals();
        // stoppingRef is intentionally NOT reset here to prevent onerror firing spuriously after close
      };

    } catch (e: any) {
      console.error(e);
      setErrorState(e?.message || 'An unexpected error occurred.');
      setLoading(false);
    }
  };

  const stopWorkout = () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setIsRecording(false);
      stopIntervals();
      return;
    }

    stoppingRef.current = true;
    wsRef.current.send(JSON.stringify({
      event: 'stop_workout',
      duration: liveDuration
    }));
  };

  if (hasPermission === null) {
    return (
      <View style={styles.loaderContainer}>
        <ActivityIndicator size="large" color="#FF3B30" />
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.loaderContainer}>
        <Text style={styles.errorText}>No access to camera</Text>
        <Text style={styles.subErrorText}>Camera permission is required to capture workout movements.</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" hidden={isRecording} />
      {!isRecording && <DrawerMenuButton style={styles.menuButton} />}

      {/* Main Camera View */}
      <View style={styles.cameraContainer}>
        <CameraPreview innerRef={cameraRef} />

        {isRecording && (
          <View style={styles.overlayContainer}>
            {/* Top Bar with Live Stats */}
            <View style={styles.liveHeader}>
              <View style={styles.badge}>
                <View style={styles.pulseRed} />
                <Text style={styles.badgeText}>LIVE</Text>
              </View>
              <Text style={styles.liveActivity}>{selectedActivity}</Text>
            </View>

            {/* Center Dashboard Overlay */}
            <View style={styles.statsCardGrid}>
              <View style={styles.overlayStatCard}>
                <Flame color="#FF3B30" size={24} />
                <Text style={styles.overlayStatVal}>{liveCalories.toFixed(1)}</Text>
                <Text style={styles.overlayStatLabel}>Calories (kcal)</Text>
              </View>

              <View style={styles.overlayStatCard}>
                <Clock color="#007AFF" size={24} />
                <Text style={styles.overlayStatVal}>
                  {Math.floor(liveDuration / 60)}:
                  {String(Math.floor(liveDuration % 60)).padStart(2, '0')}
                </Text>
                <Text style={styles.overlayStatLabel}>Duration</Text>
              </View>

              <View style={styles.overlayStatCard}>
                <Zap color="#FF9F0A" size={24} />
                <Text style={styles.overlayStatVal}>{liveIntensity}</Text>
                <Text style={styles.overlayStatLabel}>Intensity</Text>
              </View>
            </View>

            {/* No-pose warning */}
            {!poseDetected && (
              <View style={styles.poseWarning}>
                <Text style={styles.poseWarningText}>Step into frame for tracking</Text>
              </View>
            )}

            {/* Bottom Speed Panel */}
            <View style={styles.speedPanel}>
              <Activity color="#30D158" size={16} />
              <Text style={styles.speedText}>Velocity Score: {liveSpeed.toFixed(2)} m/s</Text>
            </View>
          </View>
        )}
      </View>

      {/* Control Panel (Standard selection view when not recording) */}
      {!isRecording ? (
        <ScrollView style={styles.controls} contentContainerStyle={styles.controlsContent}>
          <Text style={styles.controlsTitle}>Select Exercise</Text>

          {errorState ? (
            <Text style={styles.errorBanner}>{errorState}</Text>
          ) : null}

          <View style={styles.exerciseSelector}>
            {EXERCISES.map((act) => (
              <TouchableOpacity
                key={act}
                style={[
                  styles.exerciseChip,
                  selectedActivity === act ? styles.exerciseChipSelected : null
                ]}
                onPress={() => setSelectedActivity(act)}
              >
                <Text style={[
                  styles.exerciseChipText,
                  selectedActivity === act ? styles.exerciseChipTextSelected : null
                ]}>
                  {act}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          <TouchableOpacity 
            style={styles.startButton} 
            onPress={startWorkout}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#FFF" />
            ) : (
              <>
                <Play color="#FFF" fill="#FFF" size={18} />
                <Text style={styles.startButtonText}>Start Session</Text>
              </>
            )}
          </TouchableOpacity>
        </ScrollView>
      ) : (
        <View style={styles.stopControlPanel}>
          <TouchableOpacity style={styles.stopButton} onPress={stopWorkout}>
            <Square color="#FFF" fill="#FFF" size={20} />
            <Text style={styles.stopButtonText}>Finish Workout</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0A0A0C',
  },
  menuButton: {
    position: 'absolute',
    top: 56,
    left: 20,
    zIndex: 20,
  },
  loaderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0A0A0C',
  },
  errorText: {
    color: '#FF3B30',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subErrorText: {
    color: '#8E8E93',
    fontSize: 14,
    textAlign: 'center',
    paddingHorizontal: 40,
  },
  cameraContainer: {
    flex: 1.6,
    overflow: 'hidden',
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  overlayContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    padding: 24,
    justifyContent: 'space-between',
    backgroundColor: 'rgba(0,0,0,0.25)',
  },
  liveHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 40,
  },
  badge: {
    backgroundColor: 'rgba(255, 59, 48, 0.25)',
    borderWidth: 1,
    borderColor: '#FF3B30',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 6,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  pulseRed: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#FF3B30',
  },
  badgeText: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: 'bold',
    letterSpacing: 1,
  },
  liveActivity: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 12,
  },
  statsCardGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 10,
    marginTop: 'auto',
    marginBottom: 16,
  },
  overlayStatCard: {
    flex: 1,
    backgroundColor: 'rgba(22, 22, 26, 0.85)',
    borderRadius: 12,
    padding: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#222228',
  },
  overlayStatVal: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 6,
  },
  overlayStatLabel: {
    color: '#8E8E93',
    fontSize: 10,
    marginTop: 2,
  },
  poseWarning: {
    backgroundColor: 'rgba(255, 159, 10, 0.9)',
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 8,
    alignSelf: 'center',
    marginBottom: 8,
  },
  poseWarningText: {
    color: '#FFF',
    fontWeight: 'bold',
    fontSize: 13,
    textAlign: 'center',
  },
  speedPanel: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingVertical: 8,
    borderRadius: 8,
  },
  speedText: {
    color: '#E5E5EA',
    fontSize: 13,
    fontWeight: '500',
  },
  controls: {
    flex: 1,
    backgroundColor: '#16161A',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    borderWidth: 1,
    borderColor: '#222228',
  },
  controlsContent: {
    padding: 24,
    paddingBottom: 40,
  },
  controlsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFF',
    marginBottom: 16,
  },
  exerciseSelector: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 24,
  },
  exerciseChip: {
    backgroundColor: '#0F0F12',
    borderWidth: 1,
    borderColor: '#2C2C32',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
  },
  exerciseChipSelected: {
    backgroundColor: 'rgba(255, 59, 48, 0.15)',
    borderColor: '#FF3B30',
  },
  exerciseChipText: {
    color: '#AEAEB2',
    fontSize: 14,
    fontWeight: '500',
  },
  exerciseChipTextSelected: {
    color: '#FF3B30',
    fontWeight: 'bold',
  },
  startButton: {
    backgroundColor: '#FF3B30',
    borderRadius: 8,
    paddingVertical: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  startButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  stopControlPanel: {
    backgroundColor: '#16161A',
    padding: 24,
    paddingBottom: 40,
  },
  stopButton: {
    backgroundColor: '#FF3B30',
    borderRadius: 8,
    paddingVertical: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  stopButtonText: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
  errorBanner: {
    color: '#FF453A',
    fontSize: 13,
    fontWeight: '600',
    backgroundColor: 'rgba(255, 69, 58, 0.1)',
    borderRadius: 8,
    padding: 10,
    marginBottom: 16,
    textAlign: 'center',
  },
});
