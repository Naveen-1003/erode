import React, { useEffect, useState } from 'react';
import {
  StyleSheet,
  View,
  Text,
  ScrollView,
  ActivityIndicator,
  TouchableOpacity,
  RefreshControl,
  StatusBar,
} from 'react-native';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_URL } from '../../../services/api';
import { Flame, Clock, Dumbbell, Flower2, Wind, Play } from 'lucide-react-native';
import { DrawerMenuButton } from '../../../components/drawer-menu-button';

interface PlanExercise {
  name: string;
  sets?: number;
  reps?: number;
  duration_sec?: number;
}

interface WorkoutPlan {
  goal: string;
  focus: string;
  equipment_available: boolean;
  time_available: string;
  estimated_minutes: number;
  warmup: PlanExercise[];
  exercises: PlanExercise[];
  cooldown: PlanExercise[];
}

const TIME_LABELS: Record<string, string> = {
  '30_min': '30 Min',
  '1_hour': '1 Hour',
  '2_hour': '2 Hours',
};

const formatExercise = (ex: PlanExercise) => {
  if (ex.duration_sec) {
    return ex.sets ? `${ex.sets} x ${ex.duration_sec}s` : `${ex.duration_sec}s`;
  }
  if (ex.reps) {
    return `${ex.sets ?? 1} x ${ex.reps} reps`;
  }
  return '';
};

export default function PlanScreen() {
  const [plan, setPlan] = useState<WorkoutPlan | null>(null);
  const [showNudge, setShowNudge] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    fetchPlan();
  }, []);

  const fetchPlan = async () => {
    setError(null);
    try {
      const token = await AsyncStorage.getItem('user_token');
      if (!token) {
        router.replace('/auth/login');
        return;
      }

      const headers = { Authorization: `Bearer ${token}` };
      const [planRes, meRes] = await Promise.all([
        fetch(`${API_URL}/api/plan`, { headers }),
        fetch(`${API_URL}/api/auth/me`, { headers }),
      ]);

      if (!planRes.ok) {
        throw new Error(`Server error: ${planRes.status}`);
      }

      const planData = await planRes.json();
      setPlan(planData);

      if (meRes.ok) {
        const me = await meRes.json();
        setShowNudge(me.equipment_available === null || me.time_available === null);
      }
    } catch (e: any) {
      console.error('[Plan] fetchPlan error:', e);
      setError(e.message || 'Could not load your workout plan.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    fetchPlan();
  };

  if (loading) {
    return (
      <View style={styles.loaderContainer}>
        <ActivityIndicator size="large" color="#FF3B30" />
      </View>
    );
  }

  if (error || !plan) {
    return (
      <View style={styles.loaderContainer}>
        <Text style={styles.errorTitle}>Couldn't Load Plan</Text>
        <Text style={styles.errorMessage}>{error}</Text>
        <TouchableOpacity style={styles.retryButton} onPress={() => { setLoading(true); fetchPlan(); }}>
          <Text style={styles.retryText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.scrollContainer}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#FF3B30" colors={['#FF3B30']} />
      }
    >
      <StatusBar barStyle="light-content" />

      <View style={styles.header}>
        <DrawerMenuButton style={styles.menuButton} />
        <Text style={styles.greeting}>Today's Plan</Text>
        <Text style={styles.welcomeSubtitle}>{plan.focus}</Text>
      </View>

      {showNudge && (
        <TouchableOpacity style={styles.nudgeCard} onPress={() => router.push('/profile')}>
          <Text style={styles.nudgeText}>
            Set your equipment & time availability in your profile for a more personalized plan.
          </Text>
        </TouchableOpacity>
      )}

      <View style={styles.kpiGrid}>
        <View style={styles.kpiCard}>
          <Clock color="#BF5AF2" size={24} />
          <Text style={styles.kpiVal}>{plan.estimated_minutes}</Text>
          <Text style={styles.kpiLabel}>Est. Minutes</Text>
        </View>
        <View style={styles.kpiCard}>
          <Dumbbell color="#30D158" size={24} />
          <Text style={styles.kpiVal}>{plan.exercises.length}</Text>
          <Text style={styles.kpiLabel}>Exercises</Text>
        </View>
        <View style={styles.kpiCard}>
          <Flame color="#FF3B30" size={24} />
          <Text style={styles.kpiVal}>{TIME_LABELS[plan.time_available] ?? plan.time_available}</Text>
          <Text style={styles.kpiLabel}>Session Length</Text>
        </View>
      </View>

      <View style={styles.sectionContainer}>
        <View style={styles.sectionHeader}>
          <Wind color="#FF9F0A" size={18} />
          <Text style={styles.sectionTitle}>Warm-Up</Text>
        </View>
        {plan.warmup.map((ex, i) => (
          <View key={i} style={styles.exerciseRow}>
            <Text style={styles.exerciseName}>{ex.name}</Text>
            <Text style={styles.exerciseMeta}>{formatExercise(ex)}</Text>
          </View>
        ))}
      </View>

      <View style={styles.sectionContainer}>
        <View style={styles.sectionHeader}>
          <Dumbbell color="#FF3B30" size={18} />
          <Text style={styles.sectionTitle}>Main Workout</Text>
        </View>
        {plan.exercises.map((ex, i) => (
          <View key={i} style={styles.exerciseRow}>
            <Text style={styles.exerciseName}>{ex.name}</Text>
            <Text style={styles.exerciseMeta}>{formatExercise(ex)}</Text>
          </View>
        ))}
      </View>

      <View style={styles.sectionContainer}>
        <View style={styles.sectionHeader}>
          <Flower2 color="#30D158" size={18} />
          <Text style={styles.sectionTitle}>Cool-Down</Text>
        </View>
        {plan.cooldown.map((ex, i) => (
          <View key={i} style={styles.exerciseRow}>
            <Text style={styles.exerciseName}>{ex.name}</Text>
            <Text style={styles.exerciseMeta}>{formatExercise(ex)}</Text>
          </View>
        ))}
      </View>

      <TouchableOpacity style={styles.startButton} onPress={() => router.push('/camera')}>
        <Play color="#FFF" size={20} />
        <Text style={styles.startButtonText}>Start Live Workout</Text>
      </TouchableOpacity>
    </ScrollView>
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
    padding: 24,
  },
  errorTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FF3B30',
    marginBottom: 8,
  },
  errorMessage: {
    fontSize: 14,
    color: '#FFF',
    textAlign: 'center',
    marginBottom: 16,
  },
  retryButton: {
    backgroundColor: '#FF3B30',
    paddingHorizontal: 32,
    paddingVertical: 12,
    borderRadius: 8,
  },
  retryText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  scrollContainer: {
    padding: 24,
    paddingTop: 60,
    paddingBottom: 40,
  },
  header: {
    marginBottom: 20,
  },
  menuButton: {
    marginBottom: 16,
  },
  greeting: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#FFF',
  },
  welcomeSubtitle: {
    fontSize: 14,
    color: '#8E8E93',
    marginTop: 4,
  },
  nudgeCard: {
    backgroundColor: 'rgba(255, 59, 48, 0.1)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 59, 48, 0.3)',
    padding: 14,
    marginBottom: 20,
  },
  nudgeText: {
    color: '#FF9F9C',
    fontSize: 13,
    lineHeight: 18,
  },
  kpiGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 24,
    gap: 12,
  },
  kpiCard: {
    flex: 1,
    backgroundColor: '#16161A',
    borderRadius: 12,
    padding: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#222228',
  },
  kpiVal: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFF',
    marginTop: 8,
  },
  kpiLabel: {
    fontSize: 11,
    color: '#8E8E93',
    marginTop: 4,
    textAlign: 'center',
  },
  sectionContainer: {
    marginBottom: 24,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 14,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFF',
  },
  exerciseRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#16161A',
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#222228',
  },
  exerciseName: {
    fontSize: 15,
    fontWeight: '600',
    color: '#FFF',
    flex: 1,
  },
  exerciseMeta: {
    fontSize: 13,
    color: '#8E8E93',
    marginLeft: 12,
  },
  startButton: {
    backgroundColor: '#FF3B30',
    flexDirection: 'row',
    borderRadius: 8,
    paddingVertical: 14,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginTop: 10,
    elevation: 4,
    shadowColor: '#FF3B30',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  startButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
