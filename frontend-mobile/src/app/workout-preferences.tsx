import React, { useState } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  StatusBar,
  ScrollView,
} from 'react-native';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_URL } from '../services/api';
import { Dumbbell, Clock, ArrowRight } from 'lucide-react-native';

type TimeOption = '30_min' | '1_hour' | '2_hour';

const TIME_OPTIONS: { value: TimeOption; label: string }[] = [
  { value: '30_min', label: '30 Min' },
  { value: '1_hour', label: '1 Hour' },
  { value: '2_hour', label: '2 Hours' },
];

export default function WorkoutPreferencesScreen() {
  const [equipmentAvailable, setEquipmentAvailable] = useState<boolean | null>(null);
  const [timeAvailable, setTimeAvailable] = useState<TimeOption | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const router = useRouter();

  const handleContinue = async () => {
    if (equipmentAvailable === null || timeAvailable === null || submitting) return;
    setSubmitting(true);

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
        body: JSON.stringify({
          equipment_available: equipmentAvailable,
          time_available: timeAvailable,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to save preferences');
      }

      router.replace('/dashboard');
    } catch (e) {
      console.error('Workout preferences save error:', e);
      Alert.alert('Error', 'Could not save your preferences. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleSkip = () => {
    router.replace('/dashboard');
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        <View style={styles.header}>
          <Text style={styles.title}>Let's Personalize Your Plan</Text>
          <Text style={styles.subtitle}>
            A couple quick questions so we can tailor workouts to what you actually have to work with.
          </Text>
        </View>

        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Dumbbell color="#FF3B30" size={20} />
            <Text style={styles.cardTitle}>Do you have workout equipment?</Text>
          </View>
          <View style={styles.optionRow}>
            <TouchableOpacity
              style={[styles.optionButton, equipmentAvailable === true && styles.optionButtonSelected]}
              onPress={() => setEquipmentAvailable(true)}
              disabled={submitting}
            >
              <Text style={[styles.optionText, equipmentAvailable === true && styles.optionTextSelected]}>
                Yes
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.optionButton, equipmentAvailable === false && styles.optionButtonSelected]}
              onPress={() => setEquipmentAvailable(false)}
              disabled={submitting}
            >
              <Text style={[styles.optionText, equipmentAvailable === false && styles.optionTextSelected]}>
                No
              </Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Clock color="#FF3B30" size={20} />
            <Text style={styles.cardTitle}>How much time can you commit per session?</Text>
          </View>
          <View style={styles.optionRow}>
            {TIME_OPTIONS.map((opt) => (
              <TouchableOpacity
                key={opt.value}
                style={[styles.optionButton, timeAvailable === opt.value && styles.optionButtonSelected]}
                onPress={() => setTimeAvailable(opt.value)}
                disabled={submitting}
              >
                <Text style={[styles.optionText, timeAvailable === opt.value && styles.optionTextSelected]}>
                  {opt.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <TouchableOpacity
          style={[styles.continueButton, (equipmentAvailable === null || timeAvailable === null) && styles.continueButtonDisabled]}
          onPress={handleContinue}
          disabled={equipmentAvailable === null || timeAvailable === null || submitting}
        >
          {submitting ? (
            <ActivityIndicator color="#FFF" />
          ) : (
            <>
              <Text style={styles.continueButtonText}>Continue</Text>
              <ArrowRight color="#FFF" size={18} />
            </>
          )}
        </TouchableOpacity>

        <TouchableOpacity style={styles.skipButton} onPress={handleSkip} disabled={submitting}>
          <Text style={styles.skipButtonText}>Skip for now</Text>
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
  scrollContainer: {
    flexGrow: 1,
    padding: 24,
    paddingTop: 60,
    paddingBottom: 40,
  },
  header: {
    marginBottom: 32,
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#FFF',
  },
  subtitle: {
    fontSize: 14,
    color: '#8E8E93',
    marginTop: 8,
    lineHeight: 20,
  },
  card: {
    backgroundColor: '#16161A',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#222228',
    padding: 20,
    marginBottom: 16,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 16,
  },
  cardTitle: {
    flex: 1,
    fontSize: 15,
    fontWeight: '600',
    color: '#FFF',
  },
  optionRow: {
    flexDirection: 'row',
    backgroundColor: '#0F0F12',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#2C2C32',
    padding: 3,
    gap: 3,
  },
  optionButton: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 6,
  },
  optionButtonSelected: {
    backgroundColor: '#FF3B30',
  },
  optionText: {
    color: '#8E8E93',
    fontSize: 14,
    fontWeight: '600',
  },
  optionTextSelected: {
    color: '#FFF',
  },
  continueButton: {
    backgroundColor: '#FF3B30',
    flexDirection: 'row',
    borderRadius: 8,
    paddingVertical: 14,
    paddingHorizontal: 24,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginTop: 12,
  },
  continueButtonDisabled: {
    opacity: 0.4,
  },
  continueButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  skipButton: {
    alignItems: 'center',
    marginTop: 24,
  },
  skipButtonText: {
    color: '#8E8E93',
    fontSize: 14,
    fontWeight: '600',
    textDecorationLine: 'underline',
  },
});
