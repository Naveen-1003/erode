import React, { useEffect, useState } from 'react';
import { 
  StyleSheet, 
  View, 
  Text, 
  TextInput, 
  TouchableOpacity, 
  ActivityIndicator, 
  ScrollView, 
  Alert,
  KeyboardAvoidingView,
  Platform
} from 'react-native';
import { useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_URL } from '../../services/api';
import { User, Scale, Ruler, Calendar, LogOut, Save, Dumbbell, Clock } from 'lucide-react-native';
import { DrawerMenuButton } from '../../components/drawer-menu-button';

type TimeOption = '30_min' | '1_hour' | '2_hour';

const TIME_OPTIONS: { value: TimeOption; label: string }[] = [
  { value: '30_min', label: '30 Min' },
  { value: '1_hour', label: '1 Hour' },
  { value: '2_hour', label: '2 Hours' },
];

export default function ProfileScreen() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [age, setAge] = useState('');
  const [height, setHeight] = useState('');
  const [weight, setWeight] = useState('');
  const [gender, setGender] = useState('M');
  const [equipmentAvailable, setEquipmentAvailable] = useState<boolean | null>(null);
  const [timeAvailable, setTimeAvailable] = useState<TimeOption | null>(null);

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const router = useRouter();

  useEffect(() => {
    fetchProfile();
  }, []);

  const fetchProfile = async () => {
    try {
      const token = await AsyncStorage.getItem('user_token');
      if (!token) {
        router.replace('/auth/login');
        return;
      }

      const response = await fetch(`${API_URL}/api/auth/me`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch profile');
      }

      const data = await response.json();
      setName(data.name);
      setEmail(data.email);
      setAge(data.age.toString());
      setHeight(data.height.toString());
      setWeight(data.weight.toString());
      setGender(data.gender);
      setEquipmentAvailable(data.equipment_available ?? null);
      setTimeAvailable(data.time_available ?? null);
    } catch (e) {
      console.error(e);
      Alert.alert('Error', 'Failed to load profile details.');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!name || !age || !height || !weight) {
      Alert.alert('Error', 'Please fill out all fields.');
      return;
    }

    const parsedAge = parseInt(age);
    const parsedHeight = parseFloat(height);
    const parsedWeight = parseFloat(weight);

    if (isNaN(parsedAge) || parsedAge <= 0 || isNaN(parsedHeight) || parsedHeight <= 0 || isNaN(parsedWeight) || parsedWeight <= 0) {
      Alert.alert('Error', 'Please enter valid numerical values.');
      return;
    }

    setSaving(true);
    try {
      const token = await AsyncStorage.getItem('user_token');
      const response = await fetch(`${API_URL}/api/auth/profile`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          name,
          age: parsedAge,
          height: parsedHeight,
          weight: parsedWeight,
          gender,
          equipment_available: equipmentAvailable,
          time_available: timeAvailable,
        })
      });

      if (!response.ok) {
        throw new Error('Update failed');
      }

      const data = await response.json();
      // Update AsyncStorage
      await AsyncStorage.setItem('user_name', data.name);
      await AsyncStorage.setItem('user_age', data.age.toString());
      await AsyncStorage.setItem('user_height', data.height.toString());
      await AsyncStorage.setItem('user_weight', data.weight.toString());
      await AsyncStorage.setItem('user_gender', data.gender);

      Alert.alert('Success', 'Profile metrics updated successfully.');
    } catch (e) {
      console.error(e);
      Alert.alert('Error', 'Failed to update profile.');
    } finally {
      setSaving(false);
    }
  };

  const handleLogout = async () => {
    Alert.alert(
      'Log Out',
      'Are you sure you want to log out?',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Log Out', 
          style: 'destructive',
          onPress: async () => {
            await AsyncStorage.clear();
            router.replace('/auth/login');
          }
        }
      ]
    );
  };

  if (loading) {
    return (
      <View style={styles.loaderContainer}>
        <ActivityIndicator size="large" color="#FF3B30" />
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
    >
      <DrawerMenuButton style={styles.menuButton} />
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        <View style={styles.header}>
          <View style={styles.avatar}>
            <Text style={styles.avatarText}>{name.charAt(0).toUpperCase()}</Text>
          </View>
          <Text style={styles.name}>{name}</Text>
          <Text style={styles.email}>{email}</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Body Statistics</Text>

          <View style={styles.inputRow}>
            <Calendar color="#FF3B30" size={20} style={styles.icon} />
            <View style={styles.inputWrapper}>
              <Text style={styles.label}>Age (years)</Text>
              <TextInput
                style={styles.input}
                keyboardType="number-pad"
                value={age}
                onChangeText={setAge}
              />
            </View>
          </View>

          <View style={styles.inputRow}>
            <Ruler color="#FF3B30" size={20} style={styles.icon} />
            <View style={styles.inputWrapper}>
              <Text style={styles.label}>Height (cm)</Text>
              <TextInput
                style={styles.input}
                keyboardType="numeric"
                value={height}
                onChangeText={setHeight}
              />
            </View>
          </View>

          <View style={styles.inputRow}>
            <Scale color="#FF3B30" size={20} style={styles.icon} />
            <View style={styles.inputWrapper}>
              <Text style={styles.label}>Weight (kg)</Text>
              <TextInput
                style={styles.input}
                keyboardType="numeric"
                value={weight}
                onChangeText={setWeight}
              />
            </View>
          </View>

          <View style={styles.inputRow}>
            <User color="#FF3B30" size={20} style={styles.icon} />
            <View style={styles.inputWrapper}>
              <Text style={styles.label}>Gender</Text>
              <View style={styles.genderContainer}>
                <TouchableOpacity 
                  style={[styles.genderButton, gender === 'M' ? styles.genderButtonSelected : null]}
                  onPress={() => setGender('M')}
                >
                  <Text style={[styles.genderText, gender === 'M' ? styles.genderTextSelected : null]}>Male</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  style={[styles.genderButton, gender === 'F' ? styles.genderButtonSelected : null]}
                  onPress={() => setGender('F')}
                >
                  <Text style={[styles.genderText, gender === 'F' ? styles.genderTextSelected : null]}>Female</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Workout Preferences</Text>

          <View style={styles.inputRow}>
            <Dumbbell color="#FF3B30" size={20} style={styles.icon} />
            <View style={styles.inputWrapper}>
              <Text style={styles.label}>Equipment Available</Text>
              <View style={styles.genderContainer}>
                <TouchableOpacity
                  style={[styles.genderButton, equipmentAvailable === true ? styles.genderButtonSelected : null]}
                  onPress={() => setEquipmentAvailable(true)}
                >
                  <Text style={[styles.genderText, equipmentAvailable === true ? styles.genderTextSelected : null]}>Yes</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.genderButton, equipmentAvailable === false ? styles.genderButtonSelected : null]}
                  onPress={() => setEquipmentAvailable(false)}
                >
                  <Text style={[styles.genderText, equipmentAvailable === false ? styles.genderTextSelected : null]}>No</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>

          <View style={styles.inputRow}>
            <Clock color="#FF3B30" size={20} style={styles.icon} />
            <View style={styles.inputWrapper}>
              <Text style={styles.label}>Time Available per Session</Text>
              <View style={styles.genderContainer}>
                {TIME_OPTIONS.map((opt) => (
                  <TouchableOpacity
                    key={opt.value}
                    style={[styles.genderButton, timeAvailable === opt.value ? styles.genderButtonSelected : null]}
                    onPress={() => setTimeAvailable(opt.value)}
                  >
                    <Text style={[styles.genderText, timeAvailable === opt.value ? styles.genderTextSelected : null]}>
                      {opt.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
          </View>
        </View>

        <TouchableOpacity style={styles.saveButton} onPress={handleSave} disabled={saving}>
          {saving ? (
            <ActivityIndicator color="#FFF" />
          ) : (
            <>
              <Save color="#FFF" size={20} />
              <Text style={styles.saveButtonText}>Save Changes</Text>
            </>
          )}
        </TouchableOpacity>

        <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
          <LogOut color="#FFF" size={20} />
          <Text style={styles.logoutText}>Log Out</Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0A0A0C',
  },
  menuButton: {
    position: 'absolute',
    top: 56,
    left: 20,
    zIndex: 20,
  },
  loaderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0A0A0C',
  },
  scrollContainer: {
    padding: 24,
    paddingTop: 60,
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#FF3B30',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
    elevation: 4,
    shadowColor: '#FF3B30',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  avatarText: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#FFF',
  },
  name: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFF',
  },
  email: {
    fontSize: 14,
    color: '#8E8E93',
    marginTop: 4,
  },
  card: {
    backgroundColor: '#16161A',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: '#222228',
    marginBottom: 20,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFF',
    marginBottom: 20,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 18,
  },
  icon: {
    marginRight: 16,
    marginTop: 16,
  },
  inputWrapper: {
    flex: 1,
  },
  label: {
    fontSize: 12,
    color: '#8E8E93',
    marginBottom: 4,
  },
  input: {
    backgroundColor: '#0F0F12',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#2C2C32',
    color: '#FFF',
    fontSize: 16,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  genderContainer: {
    flexDirection: 'row',
    backgroundColor: '#0F0F12',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#2C2C32',
    padding: 2,
    height: 40,
  },
  genderButton: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 6,
  },
  genderButtonSelected: {
    backgroundColor: '#FF3B30',
  },
  genderText: {
    color: '#8E8E93',
    fontSize: 14,
    fontWeight: '600',
  },
  genderTextSelected: {
    color: '#FFF',
  },
  saveButton: {
    backgroundColor: '#FF3B30',
    flexDirection: 'row',
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10,
    gap: 8,
  },
  saveButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  logoutButton: {
    backgroundColor: '#2C2C32',
    flexDirection: 'row',
    borderRadius: 8,
    paddingVertical: 14,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    borderWidth: 1,
    borderColor: '#3A3A40',
  },
  logoutText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
