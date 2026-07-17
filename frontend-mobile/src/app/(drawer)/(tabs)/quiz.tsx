import React, { useState } from 'react';
import {
  StyleSheet,
  View,
  Text,
  Image,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  StatusBar,
  ScrollView,
} from 'react-native';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_URL } from '../../../services/api';
import { DrawerMenuButton } from '../../../components/drawer-menu-button';
import { useGoalStatus } from './_layout';

type Goal = 'fat_to_fit' | 'skinny_to_fit' | 'skinny_fat_to_fit';

const OPTIONS: { goal: Goal; label: string; image: any }[] = [
  { goal: 'fat_to_fit', label: 'Fat to Fit', image: require('@/assets/images/fat.png') },
  { goal: 'skinny_to_fit', label: 'Skinny to Fit', image: require('@/assets/images/skinny.png') },
  { goal: 'skinny_fat_to_fit', label: 'Skinny Fat to Fit', image: require('@/assets/images/skinnyfat.png') },
];

export default function QuizScreen() {
  const [submittingGoal, setSubmittingGoal] = useState<Goal | null>(null);
  const router = useRouter();
  const { refreshGoal } = useGoalStatus();

  const handleSelect = async (goal: Goal) => {
    if (submittingGoal) return;
    setSubmittingGoal(goal);

    try {
      const token = await AsyncStorage.getItem('user_token');
      if (!token) {
        router.replace('/auth/login');
        return;
      }

      const response = await fetch(`${API_URL}/api/auth/profile`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ goal }),
      });

      if (!response.ok) {
        throw new Error('Failed to save your goal');
      }

      await AsyncStorage.setItem('user_goal', goal);
      refreshGoal();
      router.replace('/workout-preferences');
    } catch (e) {
      console.error('Quiz submit error:', e);
      Alert.alert('Error', 'Could not save your goal. Please try again.');
    } finally {
      setSubmittingGoal(null);
    }
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      <View style={styles.header}>
        <DrawerMenuButton style={styles.menuButton} />
        <Text style={styles.title}>What is your goal?</Text>
        <Text style={styles.subtitle}>Pick the option that matches you best</Text>
      </View>

      <ScrollView contentContainerStyle={styles.options}>
        {OPTIONS.map((opt) => (
          <TouchableOpacity
            key={opt.goal}
            style={styles.card}
            activeOpacity={0.8}
            onPress={() => handleSelect(opt.goal)}
            disabled={!!submittingGoal}
          >
            <Image source={opt.image} style={[StyleSheet.absoluteFill, styles.cardImage]} resizeMode="cover" />
            {submittingGoal === opt.goal && (
              <View style={[StyleSheet.absoluteFill, styles.cardLoadingScrim]}>
                <ActivityIndicator size="large" color="#FFF" />
              </View>
            )}
          </TouchableOpacity>
        ))}
      </ScrollView>
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
  options: {
    padding: 24,
    paddingBottom: 40,
    gap: 16,
  },
  card: {
    width: '100%',
    aspectRatio: 4 / 5,
    borderRadius: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#222228',
    backgroundColor: '#16161A',
  },
  cardImage: {
    width: '100%',
    height: '100%',
  },
  cardLoadingScrim: {
    alignItems: 'center',
    justifyContent: 'center',
  },
});
