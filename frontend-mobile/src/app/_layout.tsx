import React from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

export default function RootLayout() {
  return (
    <>
      <StatusBar style="light" />
      <Stack 
        screenOptions={{ 
          headerShown: false, 
          contentStyle: { backgroundColor: '#0A0A0C' } 
        }}
      >
        <Stack.Screen name="index" />
        <Stack.Screen name="auth/login" />
        <Stack.Screen name="auth/register" />
        <Stack.Screen name="(drawer)" />
        <Stack.Screen name="workout-preferences" />
        <Stack.Screen name="workout/[id]" options={{ presentation: 'modal' }} />
      </Stack>
    </>
  );
}
