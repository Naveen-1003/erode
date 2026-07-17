import React, { useState } from 'react';
import { 
  StyleSheet, 
  View, 
  TextInput, 
  TouchableOpacity, 
  ActivityIndicator, 
  Text,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StatusBar
} from 'react-native';
import { useRouter } from 'expo-router';
import { API_URL } from '../../services/api';

export default function RegisterScreen() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [age, setAge] = useState('');
  const [height, setHeight] = useState('');
  const [weight, setWeight] = useState('');
  const [gender, setGender] = useState('M'); // Default to Male

  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const router = useRouter();

  const handleRegister = async () => {
    if (!name || !email || !password || !age || !height || !weight || !gender) {
      setErrorMsg('Please fill out all fields.');
      return;
    }
    
    const parsedAge = parseInt(age);
    const parsedHeight = parseFloat(height);
    const parsedWeight = parseFloat(weight);

    if (isNaN(parsedAge) || parsedAge <= 0) {
      setErrorMsg('Please enter a valid age.');
      return;
    }
    if (isNaN(parsedHeight) || parsedHeight <= 0) {
      setErrorMsg('Please enter a valid height in cm.');
      return;
    }
    if (isNaN(parsedWeight) || parsedWeight <= 0) {
      setErrorMsg('Please enter a valid weight in kg.');
      return;
    }

    setLoading(true);
    setErrorMsg('');

    try {
      const response = await fetch(`${API_URL}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          email,
          password,
          age: parsedAge,
          height: parsedHeight,
          weight: parsedWeight,
          gender
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Registration failed');
      }

      // Success, route to login
      router.replace('/auth/login');
    } catch (err: any) {
      setErrorMsg(err.message || 'Error occurred during registration.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView 
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
    >
      <StatusBar barStyle="light-content" />
      <ScrollView contentContainerStyle={styles.scrollContainer} keyboardShouldPersistTaps="handled">
        <View style={styles.header}>
          <Text style={styles.logo}>BURN<Text style={styles.logoAccent}>-EX</Text></Text>
          <Text style={styles.subtitle}>Create Your Profile</Text>
        </View>

        <View style={styles.form}>
          {errorMsg ? <Text style={styles.errorText}>{errorMsg}</Text> : null}

          <View style={styles.inputContainer}>
            <Text style={styles.label}>Full Name</Text>
            <TextInput 
              style={styles.input}
              placeholder="e.g. John Doe"
              placeholderTextColor="#555"
              value={name}
              onChangeText={setName}
            />
          </View>

          <View style={styles.inputContainer}>
            <Text style={styles.label}>Email Address</Text>
            <TextInput 
              style={styles.input}
              placeholder="e.g. john@example.com"
              placeholderTextColor="#555"
              keyboardType="email-address"
              autoCapitalize="none"
              value={email}
              onChangeText={setEmail}
            />
          </View>

          <View style={styles.inputContainer}>
            <Text style={styles.label}>Password</Text>
            <TextInput 
              style={styles.input}
              placeholder="Min. 6 characters"
              placeholderTextColor="#555"
              secureTextEntry
              value={password}
              onChangeText={setPassword}
            />
          </View>

          <View style={styles.row}>
            <View style={[styles.inputContainer, { flex: 1, marginRight: 8 }]}>
              <Text style={styles.label}>Age (yrs)</Text>
              <TextInput 
                style={styles.input}
                placeholder="25"
                placeholderTextColor="#555"
                keyboardType="number-pad"
                value={age}
                onChangeText={setAge}
              />
            </View>

            <View style={[styles.inputContainer, { flex: 1, marginLeft: 8 }]}>
              <Text style={styles.label}>Gender</Text>
              <View style={styles.genderContainer}>
                <TouchableOpacity 
                  style={[styles.genderButton, gender === 'M' ? styles.genderButtonSelected : null]}
                  onPress={() => setGender('M')}
                >
                  <Text style={[styles.genderButtonText, gender === 'M' ? styles.genderButtonTextSelected : null]}>Male</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  style={[styles.genderButton, gender === 'F' ? styles.genderButtonSelected : null]}
                  onPress={() => setGender('F')}
                >
                  <Text style={[styles.genderButtonText, gender === 'F' ? styles.genderButtonTextSelected : null]}>Female</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>

          <View style={styles.row}>
            <View style={[styles.inputContainer, { flex: 1, marginRight: 8 }]}>
              <Text style={styles.label}>Height (cm)</Text>
              <TextInput 
                style={styles.input}
                placeholder="175"
                placeholderTextColor="#555"
                keyboardType="numeric"
                value={height}
                onChangeText={setHeight}
              />
            </View>

            <View style={[styles.inputContainer, { flex: 1, marginLeft: 8 }]}>
              <Text style={styles.label}>Weight (kg)</Text>
              <TextInput 
                style={styles.input}
                placeholder="70"
                placeholderTextColor="#555"
                keyboardType="numeric"
                value={weight}
                onChangeText={setWeight}
              />
            </View>
          </View>

          <TouchableOpacity 
            style={styles.button}
            onPress={handleRegister}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#FFF" />
            ) : (
              <Text style={styles.buttonText}>Register</Text>
            )}
          </TouchableOpacity>

          <View style={styles.footer}>
            <Text style={styles.footerText}>Already have an account? </Text>
            <TouchableOpacity onPress={() => router.push('/auth/login')}>
              <Text style={styles.linkText}>Sign In</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0A0A0C',
  },
  scrollContainer: {
    flexGrow: 1,
    justifyContent: 'center',
    paddingHorizontal: 24,
    paddingVertical: 30,
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  logo: {
    fontSize: 34,
    fontWeight: 'bold',
    color: '#FFF',
    letterSpacing: 2,
  },
  logoAccent: {
    color: '#FF3B30',
  },
  subtitle: {
    color: '#8E8E93',
    fontSize: 16,
    marginTop: 4,
    letterSpacing: 1,
  },
  form: {
    backgroundColor: '#16161A',
    borderRadius: 16,
    padding: 24,
    borderWidth: 1,
    borderColor: '#222228',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 10,
  },
  errorText: {
    color: '#FF453A',
    marginBottom: 16,
    fontSize: 14,
    fontWeight: '600',
  },
  inputContainer: {
    marginBottom: 16,
  },
  label: {
    color: '#AEAEB2',
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 6,
  },
  input: {
    backgroundColor: '#0F0F12',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#2C2C32',
    paddingHorizontal: 16,
    paddingVertical: 12,
    color: '#FFF',
    fontSize: 16,
  },
  row: {
    flexDirection: 'row',
  },
  genderContainer: {
    flexDirection: 'row',
    backgroundColor: '#0F0F12',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#2C2C32',
    padding: 4,
    height: 48,
    alignItems: 'center',
  },
  genderButton: {
    flex: 1,
    height: '100%',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 6,
  },
  genderButtonSelected: {
    backgroundColor: '#FF3B30',
  },
  genderButtonText: {
    color: '#8E8E93',
    fontSize: 14,
    fontWeight: '600',
  },
  genderButtonTextSelected: {
    color: '#FFF',
  },
  button: {
    backgroundColor: '#FF3B30',
    borderRadius: 8,
    paddingVertical: 14,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10,
  },
  buttonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 20,
  },
  footerText: {
    color: '#8E8E93',
    fontSize: 14,
  },
  linkText: {
    color: '#FF3B30',
    fontSize: 14,
    fontWeight: 'bold',
  },
});
