import { Platform, NativeModules } from 'react-native';
import Constants from 'expo-constants';
import AsyncStorage from '@react-native-async-storage/async-storage';

const getBackendHost = () => {
  // Walk all known Expo Constants paths to find the Metro server host (IP:port).
  let debuggerHost: string | undefined =
    Constants.expoGoConfig?.debuggerHost ||           // Expo Go SDK 50+
    (Constants as any).manifest2?.extra?.expoGo?.debuggerHost || // dev-client manifest v2
    (Constants as any).manifest?.debuggerHost;        // Expo Go legacy SDK <49

  // In expo-dev-client builds the above paths are all undefined. Extract the Metro
  // server IP from the JS bundle URL — the most reliable source in dev builds.
  if (!debuggerHost && __DEV__) {
    const scriptURL: string | undefined = NativeModules.SourceCode?.scriptURL;
    if (scriptURL) {
      const match = scriptURL.match(/https?:\/\/([^:/]+)/);
      if (match) debuggerHost = `${match[1]}:8081`;
    }
  }

  if (Platform.OS === 'android') {
    // Always prefer the detected Metro host — check this BEFORE Constants.isDevice
    // because isDevice is unreliable in dev builds (can be false on a real device).
    if (debuggerHost) {
      const ip = debuggerHost.split(':')[0];
      // Loopback means AVD emulator; use Android's special host alias.
      if (ip === '127.0.0.1' || ip === 'localhost') return '10.0.2.2:8000';
      return `${ip}:8000`;
    }
    // No Metro host found. In release builds !isDevice reliably means AVD.
    // In dev builds isDevice is unreliable, so fall back to localhost which works
    // when `adb reverse tcp:8000 tcp:8000` is active (covers both USB and AVD).
    if (!Constants.isDevice && !__DEV__) return '10.0.2.2:8000';
    return 'localhost:8000';
  }

  // iOS / Web
  if (debuggerHost) {
    const ip = debuggerHost.split(':')[0];
    if (ip === '127.0.0.1' || ip === 'localhost') return 'localhost:8000';
    return `${ip}:8000`;
  }
  return 'localhost:8000';
};

const HOST = getBackendHost();
export const API_URL = `http://${HOST}`;
export const WS_URL = `ws://${HOST}`;

console.log(`[Burn-Ex API] Configuring endpoints. HTTP: ${API_URL}, WS: ${WS_URL}`);

const REQUEST_TIMEOUT_MS = 10000;

// Generic fetch client with JWT token attachment and timeout
export const apiRequest = async (endpoint: string, options: RequestInit = {}) => {
  const token = await AsyncStorage.getItem('user_token');

  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  } as any;

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(`${API_URL}${endpoint}`, {
      ...options,
      headers,
      signal: controller.signal,
    });

    if (!response.ok) {
      let errorDetail = 'API Request failed';
      try {
        const errorJson = await response.json();
        errorDetail = errorJson.detail || errorDetail;
      } catch (e) {}
      throw new Error(errorDetail);
    }

    return response.json();
  } catch (err: any) {
    if (err.name === 'AbortError') {
      throw new Error('Request timed out. Is the backend server running?');
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
};
