import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { Tabs } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { LayoutGrid, Camera, ClipboardList, ListChecks } from 'lucide-react-native';

const GoalStatusContext = createContext<{ hasGoal: boolean | null; refreshGoal: () => void }>({
  hasGoal: null,
  refreshGoal: () => {},
});

export function useGoalStatus() {
  return useContext(GoalStatusContext);
}

export default function TabLayout() {
  const [hasGoal, setHasGoal] = useState<boolean | null>(null);

  const refreshGoal = useCallback(() => {
    AsyncStorage.getItem('user_goal').then((goal) => setHasGoal(!!goal));
  }, []);

  useEffect(() => {
    refreshGoal();
  }, [refreshGoal]);

  return (
    <GoalStatusContext.Provider value={{ hasGoal, refreshGoal }}>
      <Tabs
        screenOptions={{
          headerShown: false,
          tabBarStyle: {
            backgroundColor: '#16161A',
            borderTopColor: '#222228',
            height: 60,
            paddingBottom: 8,
            paddingTop: 8,
          },
          tabBarActiveTintColor: '#FF3B30',
          tabBarInactiveTintColor: '#8E8E93',
          tabBarLabelStyle: {
            fontSize: 12,
            fontWeight: '600',
          },
        }}
      >
        <Tabs.Screen
          name="dashboard"
          options={{
            title: 'Dashboard',
            tabBarIcon: ({ color, size }) => <LayoutGrid color={color} size={size} />,
          }}
        />
        <Tabs.Screen
          name="plan"
          options={{
            title: 'Plan',
            tabBarIcon: ({ color, size }) => <ListChecks color={color} size={size} />,
          }}
        />
        <Tabs.Screen
          name="camera"
          options={{
            title: 'Live Workout',
            tabBarIcon: ({ color, size }) => <Camera color={color} size={size} />,
          }}
        />
        <Tabs.Screen
          name="quiz"
          options={{
            title: 'Quiz',
            tabBarIcon: ({ color, size }) => <ClipboardList color={color} size={size} />,
            // Hidden once the user has already picked a goal; visible for fresh sign-ups.
            href: hasGoal === false ? undefined : null,
          }}
        />
      </Tabs>
    </GoalStatusContext.Provider>
  );
}
