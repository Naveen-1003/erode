import os
import shutil
import tempfile
import logging
import base64
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile, Form, WebSocket, WebSocketDisconnect, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
import json
from jose import jwt, JWTError
from pydantic import BaseModel

from ..database.connection import get_db, SessionLocal
from ..database.models import Workout, User
from .auth import get_current_user, SECRET_KEY, ALGORITHM
from ..ai.activity_model import ActivityRecognizer
from ..ai.pose_model import PoseEstimator
from ..ai.intensity import IntensityEngine
from ..ai.calorie_model import CalorieRegressor
from ..services.video_processing import VideoProcessor

logger = logging.getLogger("burn_ex_prediction")

router = APIRouter(prefix="/api/predict", tags=["prediction"])

# Initialize heavy models once at startup (globals)
activity_recognizer = ActivityRecognizer()
pose_estimator = PoseEstimator()
intensity_engine = IntensityEngine()
calorie_regressor = CalorieRegressor()

# Helper to authenticate WebSockets
def get_websocket_user(token: str, db: Session) -> Optional[User]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return db.query(User).filter(User.email == email).first()
    except JWTError:
        return None

# Pydantic Schemas
class WorkoutHistoryResponse(BaseModel):
    id: int
    activity: str
    duration: float
    intensity: str
    calories: float
    created_at: Any

    model_config = {"from_attributes": True}


@router.get("/history", response_model=List[WorkoutHistoryResponse])
def get_workout_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    workouts = db.query(Workout).filter(Workout.user_id == current_user.id).order_by(Workout.created_at.desc()).all()
    return workouts

@router.get("/history/{workout_id}", response_model=WorkoutHistoryResponse)
def get_workout_detail(
    workout_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    workout = db.query(Workout).filter(Workout.id == workout_id, Workout.user_id == current_user.id).first()
    if not workout:
        raise HTTPException(status_code=404, detail="Workout not found")
    return workout

@router.post("")
def predict_workout_video(
    video: UploadFile = File(...),
    age: Optional[int] = Form(None),
    height: Optional[float] = Form(None),
    weight: Optional[float] = Form(None),
    gender: Optional[str] = Form(None),
    duration: Optional[float] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload a video. Auto-recognize the exercise and compute calorie burn.
    """
    fd, temp_file_path = tempfile.mkstemp(suffix=".mp4")
    try:
        # Save uploaded file
        with os.fdopen(fd, 'wb') as tmp:
            shutil.copyfileobj(video.file, tmp)
            
        # 1. Read video frames
        frames, metadata = VideoProcessor.read_video(temp_file_path)
        workout_duration = duration if duration is not None else metadata["duration"]
        
        if len(frames) == 0:
            raise HTTPException(status_code=400, detail="Invalid video file: No frames decoded.")

        # 2. Activity recognition (MC3-18)
        activity, confidence = activity_recognizer.predict(frames)
        
        # 3. Pose landmark extraction (MediaPipe)
        pose_history = pose_estimator.process_video_frames(frames)
        
        # 4. Intensity engine calculation
        intensity_data = intensity_engine.calculate_intensity(pose_history, fps=metadata["fps"])
        movement_score = intensity_data["movement_score"]
        intensity = intensity_data["intensity"]
        
        # Use provided parameters or fall back to user profile metrics
        u_age = age if age is not None else current_user.age
        u_height = height if height is not None else current_user.height
        u_weight = weight if weight is not None else current_user.weight
        u_gender = gender if gender is not None else current_user.gender
        
        # 5. XGBoost Calorie estimation
        calories = calorie_regressor.predict(
            activity=activity,
            age=u_age,
            height=u_height,
            weight=u_weight,
            duration_seconds=workout_duration,
            movement_score=movement_score,
            gender=u_gender
        )
        
        # 6. Save workout to database
        db_workout = Workout(
            user_id=current_user.id,
            activity=activity,
            duration=workout_duration,
            intensity=intensity,
            calories=calories
        )
        db.add(db_workout)
        db.commit()
        db.refresh(db_workout)
        
        return {
            "id": db_workout.id,
            "activity": activity,
            "confidence": round(confidence, 4),
            "intensity": intensity,
            "movement_score": movement_score,
            "calories": round(calories, 2),
            "duration": round(workout_duration, 1)
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Cleanup file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Error removing temp file {temp_file_path}: {cleanup_error}")


@router.websocket("/live")
async def live_predict_websocket(websocket: WebSocket):
    """
    Live workout tracking WebSocket.
    Expects connection query parameter 'token' for JWT auth.
    Client streams: JSON events with frame landmarks.
    """
    await websocket.accept()
    
    db = SessionLocal()
    user = None
    try:
        # 1. Handshake & Authenticate
        params = websocket.query_params
        token = params.get("token")
        if not token:
            await websocket.send_json({"error": "Unauthorized: Missing auth token"})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        user = get_websocket_user(token, db)
        if not user:
            await websocket.send_json({"error": "Unauthorized: Invalid auth token"})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        await websocket.send_json({"status": "connected", "user": user.name})
        
        # State variables for the live session
        pose_history: List[List[Dict[str, float]]] = []
        cumulative_calories = 0.0
        start_time = None
        last_frame_timestamp = None
        current_activity = "Squat" # Default or client-defined
        
        # 2. Message Loop
        while True:
            data_str = await websocket.receive_text()
            message = json.loads(data_str)
            
            event = message.get("event")
            
            if event == "start_workout":
                current_activity = message.get("activity", "Squat")
                start_time = message.get("timestamp")
                pose_history = []
                cumulative_calories = 0.0
                last_frame_timestamp = None
                await websocket.send_json({"status": "workout_started", "activity": current_activity})
                continue
                
            elif event == "frame_image":
                image_b64 = message.get("image")
                timestamp = message.get("timestamp")

                if not image_b64 or not start_time:
                    continue

                # Decode base64 JPEG → OpenCV BGR frame
                try:
                    img_bytes = base64.b64decode(image_b64)
                    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                except Exception:
                    continue

                # Run MediaPipe pose estimation on the actual camera frame
                landmarks = pose_estimator.process_frame(frame)
                active_duration = max(0.1, timestamp - start_time)

                if landmarks is None:
                    # No person detected — keep calories frozen, warn the client
                    await websocket.send_json({
                        "event": "live_update",
                        "duration": round(active_duration, 1),
                        "movement_score": 0.0,
                        "intensity": "Low",
                        "calories": round(cumulative_calories, 2),
                        "activity": current_activity,
                        "pose_detected": False,
                    })
                    continue

                pose_history.append(landmarks)
                if len(pose_history) > 30:  # 30 frames @ 1 FPS = 30-second window
                    pose_history.pop(0)

                # fps=1.0 matches the 1000ms capture interval on the frontend
                intensity_res = intensity_engine.calculate_intensity(pose_history, fps=1.0)
                movement_score = intensity_res["movement_score"]
                intensity_level = intensity_res["intensity"]

                # Incremental MET-based accumulation — XGBoost is trained on full sessions
                # and returns a fixed ~87 kcal regardless of duration for short inputs.
                # MET formula scales linearly and gives correct live estimates.
                if last_frame_timestamp is not None:
                    time_slice = min(timestamp - last_frame_timestamp, 3.0)
                    if time_slice > 0:
                        calorie_rate = calorie_regressor.met_rate_per_second(
                            activity=current_activity,
                            weight=user.weight,
                            intensity=intensity_level,
                        )
                        cumulative_calories += calorie_rate * time_slice
                last_frame_timestamp = timestamp

                await websocket.send_json({
                    "event": "live_update",
                    "duration": round(active_duration, 1),
                    "movement_score": movement_score,
                    "intensity": intensity_level,
                    "calories": round(cumulative_calories, 2),
                    "activity": current_activity,
                    "pose_detected": True,
                })

            elif event == "frame_landmarks":
                landmarks = message.get("landmarks")
                timestamp = message.get("timestamp")

                if not landmarks or not start_time:
                    continue

                pose_history.append(landmarks)
                if len(pose_history) > 150:
                    pose_history.pop(0)

                # fps=10.0 matches the 100ms interval used by older clients
                intensity_res = intensity_engine.calculate_intensity(pose_history, fps=10.0)
                movement_score = intensity_res["movement_score"]
                intensity_level = intensity_res["intensity"]

                if last_frame_timestamp is not None:
                    time_slice = min(timestamp - last_frame_timestamp, 3.0)
                    if time_slice > 0:
                        calorie_rate = calorie_regressor.met_rate_per_second(
                            activity=current_activity,
                            weight=user.weight,
                            intensity=intensity_level,
                        )
                        cumulative_calories += calorie_rate * time_slice
                last_frame_timestamp = timestamp

                await websocket.send_json({
                    "event": "live_update",
                    "duration": round(active_duration, 1),
                    "movement_score": movement_score,
                    "intensity": intensity_level,
                    "calories": round(cumulative_calories, 2),
                    "activity": current_activity,
                    "pose_detected": True,
                })
                
            elif event == "stop_workout":
                duration = message.get("duration", 0.0)
                
                # Save workout session to DB
                if duration > 1.0:
                    # Retrieve latest intensity values from final state (fps=2.0 for frame_image path)
                    intensity_res = intensity_engine.calculate_intensity(pose_history, fps=2.0)
                    final_intensity = intensity_res["intensity"]
                    
                    db_workout = Workout(
                        user_id=user.id,
                        activity=current_activity,
                        duration=duration,
                        intensity=final_intensity,
                        calories=cumulative_calories
                    )
                    db.add(db_workout)
                    db.commit()
                    db.refresh(db_workout)
                    
                    await websocket.send_json({
                        "event": "workout_saved",
                        "id": db_workout.id,
                        "activity": db_workout.activity,
                        "duration": round(db_workout.duration, 1),
                        "intensity": db_workout.intensity,
                        "calories": round(db_workout.calories, 2)
                    })
                else:
                    await websocket.send_json({"event": "workout_discarded", "reason": "Duration too short"})
                break
                
    except WebSocketDisconnect:
        logger.info("Live prediction WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        db.close()
