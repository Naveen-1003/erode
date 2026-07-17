import React, { useEffect, useState } from 'react';
import { 
  StyleSheet, 
  View, 
  Text, 
  ScrollView, 
  ActivityIndicator, 
  Dimensions, 
  TouchableOpacity,
  RefreshControl,
  StatusBar
} from 'react-native';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_URL } from '../../../services/api';
import { LineChart } from 'react-native-chart-kit';
import { Flame, Dumbbell, Clock, ChevronRight, TrendingUp } from 'lucide-react-native';
import { DrawerMenuButton } from '../../../components/drawer-menu-button';
import { formatActionLabel } from '../../../utils/format-action-label';

const screenWidth = Dimensions.get('window').width;

interface WorkoutSession {
  id: number;
  activity: string;
  duration: number;
  intensity: string;
  calories: number;
  created_at: string;
}

export default function DashboardScreen() {
  const [userName, setUserName] = useState('Athlete');
  const [weight, setWeight] = useState(70);
  const [height, setHeight] = useState(175);
  const [workouts, setWorkouts] = useState<WorkoutSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    loadUserData();
    fetchDashboardData();
  }, []);

  const loadUserData = async () => {
    try {
      const name = await AsyncStorage.getItem('user_name');
      const w = await AsyncStorage.getItem('user_weight');
      const h = await AsyncStorage.getItem('user_height');
      if (name) setUserName(name);
      if (w) setWeight(parseFloat(w));
      if (h) setHeight(parseFloat(h));
    } catch (e) {
      console.error(e);
    }
  };

  const fetchDashboardData = async () => {
    setError(null);
    try {
      const token = await AsyncStorage.getItem('user_token');
      if (!token) {
        router.replace('/auth/login');
        return;
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);

      try {
        const response = await fetch(`${API_URL}/api/predict/history`, {
          headers: { 'Authorization': `Bearer ${token}` },
          signal: controller.signal,
        });
        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        setWorkouts(data);
      } catch (fetchErr: any) {
        clearTimeout(timeoutId);
        const isAborted =
          fetchErr.name === 'AbortError' ||
          fetchErr.message?.toLowerCase().includes('cancel') ||
          fetchErr.message?.toLowerCase().includes('aborted');
        if (isAborted) {
          throw new Error('Connection timed out. Make sure the backend is running.');
        }
        throw fetchErr;
      }
    } catch (e: any) {
      console.error('[Dashboard] fetchDashboardData error:', e);
      setError(e.message || 'Could not connect to server.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    loadUserData();
    fetchDashboardData();
  };

  // Calculate stats
  const totalCalories = workouts.reduce((sum, item) => sum + item.calories, 0);
  const totalWorkouts = workouts.length;
  const totalDurationSeconds = workouts.reduce((sum, item) => sum + item.duration, 0);
  const totalDurationMins = Math.round(totalDurationSeconds / 60);

  // Compute BMI
  const heightMeters = height / 100;
  const bmi = weight / (heightMeters * heightMeters);
  const getBmiStatus = (val: number) => {
    if (val < 18.5) return 'Underweight';
    if (val < 25) return 'Healthy';
    if (val < 30) return 'Overweight';
    return 'Obese';
  };

  // Prepare chart data (past 6 sessions or past 6 days)
  const getChartData = () => {
    if (workouts.length === 0) {
      return {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        datasets: [{ data: [0, 0, 0, 0, 0, 0] }]
      };
    }

    // Take last 6 workouts and reverse to chronological order
    const recentWorkouts = [...workouts].slice(0, 6).reverse();
    const labels = recentWorkouts.map((w, index) => {
      // Return brief name or index
      const act = formatActionLabel(w.activity).split(' ')[0];
      return `${index + 1}. ${act.substring(0, 4)}`;
    });
    const calories = recentWorkouts.map(w => w.calories);

    return {
      labels,
      datasets: [{ data: calories }]
    };
  };

  const chartConfig = {
    backgroundGradientFrom: '#16161A',
    backgroundGradientTo: '#16161A',
    decimalPlaces: 0,
    color: (opacity = 1) => `rgba(255, 59, 48, ${opacity})`, // Burn-Ex Red
    labelColor: (opacity = 1) => `rgba(142, 142, 147, ${opacity})`,
    style: {
      borderRadius: 16,
    },
    propsForDots: {
      r: '5',
      strokeWidth: '2',
      stroke: '#FF3B30',
    },
  };

  if (loading) {
    return (
      <View style={styles.loaderContainer}>
        <ActivityIndicator size="large" color="#FF3B30" />
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.loaderContainer}>
        <Text style={styles.errorTitle}>Connection Failed</Text>
        <Text style={styles.errorMessage}>{error}</Text>
        <Text style={styles.errorHint}>
          Start the backend:{'\n'}uvicorn app.main:app --host 0.0.0.0 --port 8000
        </Text>
        <TouchableOpacity
          style={styles.retryButton}
          onPress={() => {
            setLoading(true);
            loadUserData();
            fetchDashboardData();
          }}
        >
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
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          tintColor="#FF3B30"
          colors={['#FF3B30']}
        />
      }
    >
      <StatusBar barStyle="light-content" />

      {/* Header */}
      <View style={styles.header}>
        <DrawerMenuButton style={styles.menuButton} />
        <Text style={styles.greeting}>Hello, {userName} 👋</Text>
        <Text style={styles.welcomeSubtitle}>Ready to burn some calories today?</Text>
      </View>

      {/* KPI Cards Grid */}
      <View style={styles.kpiGrid}>
        <View style={styles.kpiCard}>
          <Flame color="#FF3B30" size={24} />
          <Text style={styles.kpiVal}>{Math.round(totalCalories)}</Text>
          <Text style={styles.kpiLabel}>kcal Burned</Text>
        </View>

        <View style={styles.kpiCard}>
          <Dumbbell color="#30D158" size={24} />
          <Text style={styles.kpiVal}>{totalWorkouts}</Text>
          <Text style={styles.kpiLabel}>Workouts</Text>
        </View>

        <View style={styles.kpiCard}>
          <Clock color="#BF5AF2" size={24} />
          <Text style={styles.kpiVal}>{totalDurationMins}</Text>
          <Text style={styles.kpiLabel}>Minutes Active</Text>
        </View>
      </View>

      {/* Chart Section */}
      <View style={styles.sectionContainer}>
        <View style={styles.sectionHeader}>
          <TrendingUp color="#FF3B30" size={18} />
          <Text style={styles.sectionTitle}>Calorie Progression</Text>
        </View>
        <View style={styles.chartWrapper}>
          <LineChart
            data={getChartData()}
            width={screenWidth - 48}
            height={200}
            chartConfig={chartConfig}
            bezier
            style={styles.chart}
          />
        </View>
      </View>

      {/* Physiological Summary */}
      <View style={styles.sectionContainer}>
        <Text style={styles.sectionTitle}>Your Metrics & Health</Text>
        <View style={styles.healthCard}>
          <View style={styles.healthItem}>
            <Text style={styles.healthLabel}>Weight</Text>
            <Text style={styles.healthVal}>{weight} kg</Text>
          </View>
          <View style={styles.healthDivider} />
          <View style={styles.healthItem}>
            <Text style={styles.healthLabel}>Height</Text>
            <Text style={styles.healthVal}>{height} cm</Text>
          </View>
          <View style={styles.healthDivider} />
          <View style={styles.healthItem}>
            <Text style={styles.healthLabel}>BMI</Text>
            <Text style={styles.healthVal}>{bmi.toFixed(1)}</Text>
            <Text style={styles.healthSubVal}>{getBmiStatus(bmi)}</Text>
          </View>
        </View>
      </View>

      {/* Recent Workouts */}
      <View style={styles.sectionContainer}>
        <View style={styles.sectionHeaderRow}>
          <Text style={styles.sectionTitle}>Recent Sessions</Text>
          <TouchableOpacity onPress={() => router.push('/history')}>
            <Text style={styles.viewAllText}>View All</Text>
          </TouchableOpacity>
        </View>

        {workouts.length === 0 ? (
          <View style={styles.emptyWorkouts}>
            <Text style={styles.emptyText}>No sessions recorded. Start your first workout!</Text>
          </View>
        ) : (
          workouts.slice(0, 3).map((w) => (
            <TouchableOpacity 
              key={w.id} 
              style={styles.recentItem}
              onPress={() => router.push(`/workout/${w.id}`)}
            >
              <View style={styles.recentIconBox}>
                <Dumbbell color="#FF3B30" size={18} />
              </View>
              <View style={styles.recentDetails}>
                <Text style={styles.recentTitle}>{formatActionLabel(w.activity)}</Text>
                <Text style={styles.recentMeta}>{Math.round(w.duration / 60)} min • {w.intensity} Intensity</Text>
              </View>
              <View style={styles.recentRight}>
                <Text style={styles.recentCalories}>+{Math.round(w.calories)}</Text>
                <Text style={styles.recentUnit}>kcal</Text>
              </View>
              <ChevronRight color="#444" size={16} />
            </TouchableOpacity>
          ))
        )}
      </View>

      {/* Quick Start Button */}
      <TouchableOpacity
        style={styles.quickStartButton}
        onPress={() => router.push('/camera')}
      >
        <Dumbbell color="#FFF" size={20} />
        <Text style={styles.quickStartText}>Start Live Workout</Text>
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
  errorHint: {
    fontSize: 12,
    color: '#8E8E93',
    textAlign: 'center',
    fontFamily: 'monospace',
    backgroundColor: '#16161A',
    padding: 12,
    borderRadius: 8,
    marginBottom: 24,
    lineHeight: 20,
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
    marginBottom: 24,
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
    fontSize: 20,
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
  sectionHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 14,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFF',
  },
  viewAllText: {
    color: '#FF3B30',
    fontSize: 14,
    fontWeight: '600',
  },
  chartWrapper: {
    backgroundColor: '#16161A',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#222228',
    padding: 12,
    alignItems: 'center',
  },
  chart: {
    marginVertical: 4,
    borderRadius: 16,
  },
  healthCard: {
    flexDirection: 'row',
    backgroundColor: '#16161A',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: '#222228',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  healthItem: {
    flex: 1,
    alignItems: 'center',
  },
  healthLabel: {
    fontSize: 12,
    color: '#8E8E93',
    marginBottom: 4,
  },
  healthVal: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFF',
  },
  healthSubVal: {
    fontSize: 11,
    color: '#30D158',
    fontWeight: '600',
    marginTop: 2,
  },
  healthDivider: {
    width: 1,
    height: 30,
    backgroundColor: '#2C2C32',
  },
  emptyWorkouts: {
    backgroundColor: '#16161A',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#222228',
  },
  emptyText: {
    color: '#8E8E93',
    fontSize: 14,
    textAlign: 'center',
  },
  recentItem: {
    flexDirection: 'row',
    backgroundColor: '#16161A',
    borderRadius: 12,
    padding: 14,
    alignItems: 'center',
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#222228',
  },
  recentIconBox: {
    width: 36,
    height: 36,
    borderRadius: 8,
    backgroundColor: 'rgba(255, 59, 48, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  recentDetails: {
    flex: 1,
  },
  recentTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFF',
  },
  recentMeta: {
    fontSize: 12,
    color: '#8E8E93',
    marginTop: 2,
  },
  recentRight: {
    alignItems: 'flex-end',
    marginRight: 10,
  },
  recentCalories: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FF3B30',
  },
  recentUnit: {
    fontSize: 10,
    color: '#8E8E93',
  },
  quickStartButton: {
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
  quickStartText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
