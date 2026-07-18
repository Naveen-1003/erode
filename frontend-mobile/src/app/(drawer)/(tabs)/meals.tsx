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
import { Flame, Drumstick, Wheat, Droplet, Coffee, Sun, Cookie, Moon } from 'lucide-react-native';
import { DrawerMenuButton } from '../../../components/drawer-menu-button';

interface Meal {
  name: string;
  calories: number;
  protein_g: number;
  carbs_g: number;
  fat_g: number;
  portion_label: string;
}

interface MealPlan {
  goal: string;
  food_preference: string;
  target_calories: number;
  target_protein_g: number;
  target_carbs_g: number;
  target_fat_g: number;
  meals: {
    breakfast: Meal;
    lunch: Meal;
    snack: Meal;
    dinner: Meal;
  };
  totals: {
    calories: number;
    protein_g: number;
    carbs_g: number;
    fat_g: number;
  };
}

const GOAL_LABELS: Record<string, string> = {
  fat_to_fit: 'Fat Loss Focus',
  skinny_to_fit: 'Muscle Gain Focus',
  skinny_fat_to_fit: 'Balanced Recomp',
};

const SLOT_ORDER: { key: keyof MealPlan['meals']; label: string; Icon: typeof Coffee }[] = [
  { key: 'breakfast', label: 'Breakfast', Icon: Coffee },
  { key: 'lunch', label: 'Lunch', Icon: Sun },
  { key: 'snack', label: 'Snack', Icon: Cookie },
  { key: 'dinner', label: 'Dinner', Icon: Moon },
];

export default function MealsScreen() {
  const [plan, setPlan] = useState<MealPlan | null>(null);
  const [showNudge, setShowNudge] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    fetchMealPlan();
  }, []);

  const fetchMealPlan = async () => {
    setError(null);
    try {
      const token = await AsyncStorage.getItem('user_token');
      if (!token) {
        router.replace('/auth/login');
        return;
      }

      const headers = { Authorization: `Bearer ${token}` };
      const [mealRes, meRes] = await Promise.all([
        fetch(`${API_URL}/api/plan/meal`, { headers }),
        fetch(`${API_URL}/api/auth/me`, { headers }),
      ]);

      if (!mealRes.ok) {
        throw new Error(`Server error: ${mealRes.status}`);
      }

      const mealData = await mealRes.json();
      setPlan(mealData);

      if (meRes.ok) {
        const me = await meRes.json();
        setShowNudge(!me.food_preference);
      }
    } catch (e: any) {
      console.error('[Meals] fetchMealPlan error:', e);
      setError(e.message || 'Could not load your meal plan.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    fetchMealPlan();
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
        <Text style={styles.errorTitle}>Couldn't Load Meal Plan</Text>
        <Text style={styles.errorMessage}>{error}</Text>
        <TouchableOpacity style={styles.retryButton} onPress={() => { setLoading(true); fetchMealPlan(); }}>
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
        <Text style={styles.greeting}>Today's Meals</Text>
        <Text style={styles.welcomeSubtitle}>{GOAL_LABELS[plan.goal] ?? plan.goal}</Text>
      </View>

      {showNudge && (
        <TouchableOpacity style={styles.nudgeCard} onPress={() => router.push('/profile')}>
          <Text style={styles.nudgeText}>
            Set your food preference in your profile for a more personalized meal plan.
          </Text>
        </TouchableOpacity>
      )}

      <View style={styles.kpiGrid}>
        <View style={styles.kpiCard}>
          <Flame color="#FF3B30" size={22} />
          <Text style={styles.kpiVal}>{plan.target_calories}</Text>
          <Text style={styles.kpiLabel}>Calories</Text>
        </View>
        <View style={styles.kpiCard}>
          <Drumstick color="#30D158" size={22} />
          <Text style={styles.kpiVal}>{plan.target_protein_g}g</Text>
          <Text style={styles.kpiLabel}>Protein</Text>
        </View>
        <View style={styles.kpiCard}>
          <Wheat color="#FF9F0A" size={22} />
          <Text style={styles.kpiVal}>{plan.target_carbs_g}g</Text>
          <Text style={styles.kpiLabel}>Carbs</Text>
        </View>
        <View style={styles.kpiCard}>
          <Droplet color="#BF5AF2" size={22} />
          <Text style={styles.kpiVal}>{plan.target_fat_g}g</Text>
          <Text style={styles.kpiLabel}>Fat</Text>
        </View>
      </View>

      {SLOT_ORDER.map(({ key, label, Icon }) => {
        const meal = plan.meals[key];
        if (!meal) return null;
        return (
          <View key={key} style={styles.sectionContainer}>
            <View style={styles.sectionHeader}>
              <Icon color="#FF3B30" size={18} />
              <Text style={styles.sectionTitle}>{label}</Text>
            </View>
            <View style={styles.mealCard}>
              <View style={styles.mealHeaderRow}>
                <Text style={styles.mealName}>{meal.name}</Text>
                <Text style={styles.mealCalories}>{meal.calories} kcal</Text>
              </View>
              <Text style={styles.mealPortion}>{meal.portion_label}</Text>
              <View style={styles.macroRow}>
                <Text style={styles.macroText}>P {meal.protein_g}g</Text>
                <Text style={styles.macroText}>C {meal.carbs_g}g</Text>
                <Text style={styles.macroText}>F {meal.fat_g}g</Text>
              </View>
            </View>
          </View>
        );
      })}

      <View style={styles.totalsCard}>
        <Text style={styles.totalsTitle}>Daily Totals</Text>
        <Text style={styles.totalsText}>
          {plan.totals.calories} kcal · P {plan.totals.protein_g}g · C {plan.totals.carbs_g}g · F {plan.totals.fat_g}g
        </Text>
      </View>
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
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 24,
    gap: 12,
  },
  kpiCard: {
    width: '47%',
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
    marginBottom: 18,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 10,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFF',
  },
  mealCard: {
    backgroundColor: '#16161A',
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: '#222228',
  },
  mealHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  mealName: {
    fontSize: 15,
    fontWeight: '600',
    color: '#FFF',
    flex: 1,
    marginRight: 12,
  },
  mealCalories: {
    fontSize: 14,
    fontWeight: '600',
    color: '#FF9F0A',
  },
  mealPortion: {
    fontSize: 12,
    color: '#8E8E93',
    marginTop: 4,
  },
  macroRow: {
    flexDirection: 'row',
    gap: 16,
    marginTop: 10,
  },
  macroText: {
    fontSize: 12,
    color: '#AEAEB2',
    fontWeight: '500',
  },
  totalsCard: {
    backgroundColor: '#16161A',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: '#222228',
    marginTop: 6,
  },
  totalsTitle: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#FFF',
    marginBottom: 6,
  },
  totalsText: {
    fontSize: 13,
    color: '#8E8E93',
  },
});
