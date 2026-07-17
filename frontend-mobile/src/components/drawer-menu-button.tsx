import React from 'react';
import { StyleSheet, StyleProp, TouchableOpacity, ViewStyle } from 'react-native';
import { useNavigation } from 'expo-router';
import { Menu } from 'lucide-react-native';

export function DrawerMenuButton({ style }: { style?: StyleProp<ViewStyle> }) {
  const navigation = useNavigation();

  return (
    <TouchableOpacity
      style={[styles.button, style]}
      onPress={() => navigation.dispatch({ type: 'TOGGLE_DRAWER' })}
      hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
    >
      <Menu color="#FFF" size={20} />
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(22, 22, 26, 0.85)',
    borderWidth: 1,
    borderColor: '#222228',
  },
});
