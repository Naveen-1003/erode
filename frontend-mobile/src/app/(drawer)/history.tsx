import React, { useEffect, useState } from 'react';
import { 
  StyleSheet, 
  View, 
  Text, 
  FlatList, 
  TouchableOpacity, 
  ActivityIndicator, 
  RefreshControl,
  StatusBar
} from 'react-native';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_URL } from '../../services/api';
import { ChevronRight, Calendar, Flame, Clock } from 'lucide-react-native';
import { DrawerMenuButton } from '../../components/drawer-menu-button';

interface WorkoutSession {
  id: number;
  activity: string;
  duration: number; // in seconds
  intensity: string;
  calories: number;
  created_at: string;
}

export default function HistoryScreen() {
  const [workouts, setWorkouts] = useState<WorkoutSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const router = useRouter();

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const token = await AsyncStorage.getItem('user_token');
      if (!token) {
        router.replace('/auth/login');
        return;
      }

      const response = await fetch(`${API_URL}/api/predict/history`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (!response.ok) {
        throw new Error('Failed to load history');
      }

      const data = await response.json();
      setWorkouts(data);
    } catch (e) {
      console.error('History fetch error:', e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    fetchHistory();
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
        month: 'short', 
        day: 'numeric', 
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (e) {
      return dateStr;
    }
  };

  const getIntensityStyle = (intensity: string) => {
    const norm = intensity.toLowerCase();
    if (norm === 'high') {
      return { container: styles.intensityHigh, text: styles.intensityHighText };
    } else if (norm === 'medium') {
      return { container: styles.intensityMed, text: styles.intensityMedText };
    }
    return { container: styles.intensityLow, text: styles.intensityLowText };
  };

  const renderWorkoutItem = ({ item }: { item: WorkoutSession }) => {
    const intensityStyle = getIntensityStyle(item.intensity);
    return (
      <TouchableOpacity 
        style={styles.card}
        onPress={() => router.push(`/workout/${item.id}`)}
      >
        <View style={styles.cardHeader}>
          <Text style={styles.activityTitle}>{item.activity}</Text>
          <View style={[styles.intensityBadge, intensityStyle.container]}>
            <Text style={[styles.intensityText, intensityStyle.text]}>{item.intensity}</Text>
          </View>
        </View>

        <View style={styles.cardDetails}>
          <View style={styles.detailRow}>
            <Clock color="#8E8E93" size={16} />
            <Text style={styles.detailVal}>{formatDuration(item.duration)}</Text>
          </View>

          <View style={styles.detailRow}>
            <Flame color="#FF3B30" size={16} />
            <Text style={styles.detailVal}>{Math.round(item.calories)} kcal</Text>
          </View>
        </View>

        <View style={styles.cardFooter}>
          <View style={styles.dateRow}>
            <Calendar color="#8E8E93" size={14} />
            <Text style={styles.dateText}>{formatDate(item.created_at)}</Text>
          </View>
          <ChevronRight color="#555" size={18} />
        </View>
      </TouchableOpacity>
    );
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      <View style={styles.header}>
        <DrawerMenuButton style={styles.menuButton} />
        <Text style={styles.title}>Workout History</Text>
        <Text style={styles.subtitle}>Track your performance progression</Text>
      </View>

      {loading ? (
        <View style={styles.loaderContainer}>
          <ActivityIndicator size="large" color="#FF3B30" />
        </View>
      ) : workouts.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyTitle}>No Workouts Yet</Text>
          <Text style={styles.emptySubtitle}>Completed live workouts will appear here.</Text>
        </View>
      ) : (
        <FlatList
          data={workouts}
          keyExtractor={(item) => item.id.toString()}
          renderItem={renderWorkoutItem}
          contentContainerStyle={styles.listContainer}
          refreshControl={
            <RefreshControl 
              refreshing={refreshing} 
              onRefresh={onRefresh}
              tintColor="#FF3B30"
              colors={['#FF3B30']}
            />
          }
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0A0A0C',
  },
  header: {
    paddingHorizontal: 24,
    paddingTop: 60,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#16161A',
  },
  menuButton: {
    marginBottom: 16,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#FFF',
  },
  subtitle: {
    fontSize: 14,
    color: '#8E8E93',
    marginTop: 4,
  },
  loaderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFF',
    marginBottom: 8,
  },
  emptySubtitle: {
    fontSize: 14,
    color: '#8E8E93',
    textAlign: 'center',
    lineHeight: 20,
  },
  listContainer: {
    padding: 24,
    paddingBottom: 40,
  },
  card: {
    backgroundColor: '#16161A',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#222228',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  activityTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFF',
  },
  intensityBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  intensityText: {
    fontSize: 12,
    fontWeight: 'bold',
  },
  intensityHigh: {
    backgroundColor: 'rgba(255, 69, 58, 0.15)',
  },
  intensityHighText: {
    color: '#FF453A',
  },
  intensityMed: {
    backgroundColor: 'rgba(255, 159, 10, 0.15)',
  },
  intensityMedText: {
    color: '#FF9F0A',
  },
  intensityLow: {
    backgroundColor: 'rgba(48, 209, 88, 0.15)',
  },
  intensityLowText: {
    color: '#30D158',
  },
  cardDetails: {
    flexDirection: 'row',
    gap: 20,
    marginBottom: 12,
  },
  detailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  detailVal: {
    color: '#E5E5EA',
    fontSize: 14,
    fontWeight: '500',
  },
  cardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: '#222228',
    paddingTop: 12,
    marginTop: 4,
  },
  dateRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  dateText: {
    fontSize: 12,
    color: '#8E8E93',
  },
});
