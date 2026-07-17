import React, { useEffect, useState } from 'react';
import { 
  StyleSheet, 
  View, 
  Text, 
  ActivityIndicator, 
  TouchableOpacity, 
  ScrollView,
  StatusBar
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_URL } from '../../services/api';
import { Flame, Clock, Zap, ShieldAlert, ArrowLeft, HeartPulse } from 'lucide-react-native';

interface WorkoutSession {
  id: number;
  activity: string;
  duration: number;
  intensity: string;
  calories: number;
  created_at: string;
}

export default function WorkoutDetailScreen() {
  const { id } = useLocalSearchParams();
  const [workout, setWorkout] = useState<WorkoutSession | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    fetchDetail();
  }, [id]);

  const fetchDetail = async () => {
    try {
      const token = await AsyncStorage.getItem('user_token');
      if (!token) {
        router.replace('/auth/login');
        return;
      }

      const response = await fetch(`${API_URL}/api/predict/history/${id}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (!response.ok) {
        throw new Error('Workout details not found');
      }

      const data = await response.json();
      setWorkout(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    if (mins > 0) {
      return `${mins}m ${secs}s`;
    }
    return `${secs}s`;
  };

  const formatDate = (dateStr: string) => {
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString(undefined, { 
        weekday: 'long',
        month: 'long', 
        day: 'numeric', 
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (e) {
      return dateStr;
    }
  };

  const getPersonalizedFeedback = (w: WorkoutSession) => {
    const act = w.activity.toLowerCase();
    const kcal = w.calories;
    const durMins = w.duration / 60;
    
    if (w.intensity.toLowerCase() === 'high') {
      return `Outstanding effort! You maintained a High movement velocity during this ${w.activity} session. This triggers a high cardiorespiratory demand, maximizing calorie burn (${Math.round(kcal)} kcal) and boosting post-exercise oxygen consumption (EPOC). Make sure to prioritize hydration and consume a protein-rich meal within 2 hours for muscle recovery.`;
    } else if (w.intensity.toLowerCase() === 'medium') {
      return `Solid conditioning! You kept a consistent Medium intensity pace for ${Math.round(durMins)} minutes. This is the optimal aerobic zone for improving metabolic health, burning fat, and building muscular endurance. Great job logging this session!`;
    } else {
      return `Great recovery session! Low-intensity movement is perfect for active recovery, stimulating blood circulation to help flush lactic acid out of muscles without placing excessive stress on your joints. This type of training helps maintain your weekly consistency.`;
    }
  };

  if (loading) {
    return (
      <View style={styles.loaderContainer}>
        <ActivityIndicator size="large" color="#FF3B30" />
      </View>
    );
  }

  if (!workout) {
    return (
      <View style={styles.loaderContainer}>
        <Text style={styles.errorText}>Session details could not be found.</Text>
        <TouchableOpacity style={styles.backBtn} onPress={() => router.back()}>
          <ArrowLeft color="#FFF" size={16} />
          <Text style={styles.backBtnText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.iconButton} onPress={() => router.back()}>
          <ArrowLeft color="#FFF" size={24} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Session Summary</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView contentContainerStyle={styles.scrollContainer}>
        {/* Activity Name & Date */}
        <View style={styles.titleSection}>
          <Text style={styles.activityName}>{workout.activity}</Text>
          <Text style={styles.date}>{formatDate(workout.created_at)}</Text>
        </View>

        {/* Highlight Metrics */}
        <View style={styles.metricsBox}>
          <View style={styles.metricItem}>
            <View style={[styles.iconCircle, { backgroundColor: 'rgba(255, 59, 48, 0.1)' }]}>
              <Flame color="#FF3B30" size={28} />
            </View>
            <Text style={styles.metricVal}>{Math.round(workout.calories)}</Text>
            <Text style={styles.metricLabel}>Total Calories (kcal)</Text>
          </View>

          <View style={styles.metricItem}>
            <View style={[styles.iconCircle, { backgroundColor: 'rgba(0, 122, 255, 0.1)' }]}>
              <Clock color="#007AFF" size={28} />
            </View>
            <Text style={styles.metricVal}>{formatDuration(workout.duration)}</Text>
            <Text style={styles.metricLabel}>Workout Duration</Text>
          </View>
        </View>

        {/* Intensity Metric */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Biomechanics & Intensity</Text>
          <View style={styles.intensityRow}>
            <Zap color="#FF9F0A" size={20} />
            <Text style={styles.intensityLabel}>Calculated Intensity: </Text>
            <Text style={[
              styles.intensityVal, 
              workout.intensity.toLowerCase() === 'high' ? styles.colorHigh : 
              workout.intensity.toLowerCase() === 'medium' ? styles.colorMed : styles.colorLow
            ]}>
              {workout.intensity}
            </Text>
          </View>
          <Text style={styles.intensityDesc}>
            Intensity is determined dynamically from your joint velocity coordinates captured during exercise.
          </Text>
        </View>

        {/* Personalized Coach Feedback */}
        <View style={styles.card}>
          <View style={styles.feedbackHeader}>
            <HeartPulse color="#FF3B30" size={20} />
            <Text style={styles.cardTitle}>Burn-Ex Coach Feedback</Text>
          </View>
          <Text style={styles.feedbackText}>
            {getPersonalizedFeedback(workout)}
          </Text>
        </View>

        <TouchableOpacity 
          style={styles.doneButton}
          onPress={() => router.replace('/dashboard')}
        >
          <Text style={styles.doneButtonText}>Done</Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0A0A0C',
  },
  loaderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0A0A0C',
  },
  errorText: {
    color: '#FF3B30',
    fontSize: 16,
    marginBottom: 20,
  },
  backBtn: {
    backgroundColor: '#FF3B30',
    flexDirection: 'row',
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 10,
    alignItems: 'center',
    gap: 8,
  },
  backBtnText: {
    color: '#FFF',
    fontWeight: 'bold',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 50,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#16161A',
  },
  iconButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFF',
  },
  scrollContainer: {
    padding: 24,
    paddingBottom: 40,
  },
  titleSection: {
    alignItems: 'center',
    marginBottom: 24,
  },
  activityName: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#FFF',
  },
  date: {
    fontSize: 14,
    color: '#8E8E93',
    marginTop: 6,
  },
  metricsBox: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 16,
    marginBottom: 24,
  },
  metricItem: {
    flex: 1,
    backgroundColor: '#16161A',
    borderRadius: 16,
    padding: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#222228',
  },
  iconCircle: {
    width: 50,
    height: 50,
    borderRadius: 25,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  metricVal: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFF',
  },
  metricLabel: {
    fontSize: 11,
    color: '#8E8E93',
    marginTop: 6,
    textAlign: 'center',
  },
  card: {
    backgroundColor: '#16161A',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: '#222228',
    marginBottom: 20,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFF',
    marginBottom: 12,
  },
  intensityRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  intensityLabel: {
    color: '#AEAEB2',
    fontSize: 15,
    marginLeft: 8,
  },
  intensityVal: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  colorHigh: {
    color: '#FF453A',
  },
  colorMed: {
    color: '#FF9F0A',
  },
  colorLow: {
    color: '#30D158',
  },
  intensityDesc: {
    color: '#8E8E93',
    fontSize: 12,
    lineHeight: 18,
    marginTop: 4,
  },
  feedbackHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12,
  },
  feedbackText: {
    color: '#E5E5EA',
    fontSize: 14,
    lineHeight: 22,
  },
  doneButton: {
    backgroundColor: '#2C2C32',
    borderRadius: 8,
    paddingVertical: 14,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10,
    borderWidth: 1,
    borderColor: '#3A3A40',
  },
  doneButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
