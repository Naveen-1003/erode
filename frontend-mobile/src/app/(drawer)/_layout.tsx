import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Drawer } from 'expo-router/drawer';
import { DrawerContentScrollView, DrawerItemList } from 'expo-router/drawer';
import { LayoutGrid, History, User, Flame } from 'lucide-react-native';

function CustomDrawerContent(props: any) {
  return (
    <DrawerContentScrollView {...props} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <View style={styles.logoBadge}>
          <Flame color="#FFF" size={22} />
        </View>
        <Text style={styles.appName}>Burn-Ex</Text>
      </View>
      <DrawerItemList {...props} />
    </DrawerContentScrollView>
  );
}

export default function DrawerLayout() {
  return (
    <Drawer
      drawerContent={CustomDrawerContent}
      screenOptions={{
        headerShown: false,
        drawerType: 'front',
        drawerStyle: { backgroundColor: '#16161A', width: 260 },
        drawerActiveTintColor: '#FF3B30',
        drawerInactiveTintColor: '#8E8E93',
        drawerActiveBackgroundColor: 'rgba(255, 59, 48, 0.12)',
        drawerLabelStyle: { fontSize: 15, fontWeight: '600', marginLeft: -8 },
        overlayColor: 'rgba(0, 0, 0, 0.6)',
      }}
    >
      <Drawer.Screen
        name="(tabs)"
        options={{
          drawerLabel: 'Home',
          drawerIcon: ({ color, size }) => (
            <LayoutGrid color={color as string} size={size} />
          ),
        }}
      />
      <Drawer.Screen
        name="history"
        options={{
          drawerLabel: 'History',
          drawerIcon: ({ color, size }) => (
            <History color={color as string} size={size} />
          ),
        }}
      />
      <Drawer.Screen
        name="profile"
        options={{
          drawerLabel: 'Profile',
          drawerIcon: ({ color, size }) => (
            <User color={color as string} size={size} />
          ),
        }}
      />
    </Drawer>
  );
}

const styles = StyleSheet.create({
  content: {
    paddingTop: 0,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 24,
    marginBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#222228',
  },
  logoBadge: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: '#FF3B30',
    alignItems: 'center',
    justifyContent: 'center',
  },
  appName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFF',
  },
});
