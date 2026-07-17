# Burn-Ex Calorie Tracker

AI-powered fitness app with real-time pose estimation and calorie tracking. FastAPI backend + Expo React Native mobile app.

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- Android SDK / ADB (for USB debugging)
- A physical Android device or AVD emulator

---

## Backend

```bash
cd backend

# Install dependencies
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# Download ML model weights (first-time setup)
# Note: This step also clones YOWOv2 and fetches its weights.
python models/download_weights.py

# Run the server (accessible from all network interfaces)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

API documentation is available at:

`http://localhost:8000/docs`

> **SQLite fallback:** If PostgreSQL is not configured, the app automatically uses SQLite—no extra setup needed for local development.

---

## Frontend (Metro Dev Server)

Run this after the backend is already running.

```bash
cd frontend-mobile

# Install dependencies
npm install

# Start Metro bundler (with cache cleared)
npx expo start --clear
```

Scan the QR code in the terminal with your device's camera (using the Expo Dev Client app) to open the app.

---

## Build Dev Client for USB Debugging

You only need to do this **once**, or whenever you add/remove native dependencies.

### 1. Connect your Android device

Enable **Developer Options** and **USB Debugging** on your device, then plug it in.

```bash
# Verify the device is detected
adb devices
```

### 2. Build and install the Dev Client APK

```bash
cd frontend-mobile

npx expo run:android
```

This compiles the native Android project and installs the Dev Client APK directly onto your connected device.

The Metro bundler starts automatically when the build finishes.

### 3. Forward the backend port over USB

Run this every time you start a new USB debugging session:

```bash
adb reverse tcp:8000 tcp:8000
```

This tunnels `localhost:8000` on your device to `localhost:8000` on your PC, so the app can reach the backend over the USB cable without needing Wi-Fi.

### 4. Start Metro (Subsequent Runs)

After the APK is installed, you only need to start Metro—no need to rebuild.

```bash
cd frontend-mobile

npx expo start --clear
```

Open the app on your device and it will connect to Metro automatically.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | JWT signing secret for the backend | `dev-secret` |
| `DATABASE_URL` | PostgreSQL connection string | SQLite fallback |
