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
from ..database.models import Workout, User, WorkoutFrame
from .auth import get_current_user, SECRET_KEY, ALGORITHM
from ..ai.activity_model import ActivityRecognizer
from ..ai.pose_model import PoseEstimator
from ..ai.intensity import IntensityEngine
from ..ai.calorie_model import CalorieRegressor
from ..ai.pose_energy_model import PoseEnergyEstimator
from ..ai.yowo_action_model import YowoActionDetector
from ..services.video_processing import VideoProcessor

logger = logging.getLogger("burn_ex_prediction")

router = APIRouter(prefix="/api/predict", tags=["prediction"])

# Initialize heavy models once at startup (globals)
activity_recognizer = ActivityRecognizer()
pose_estimator = PoseEstimator()
intensity_engine = IntensityEngine()
calorie_regressor = CalorieRegressor()
# End-to-end pose->energy regressor. Fallback calorie source when YOWOv2 can't
# confidently name an action for the current window (see yowo_action_detector).
pose_energy_estimator = PoseEnergyEstimator()
# Pretrained action detector (YOWOv2 + AVA, 80 everyday actions incl. wave,
# handshake, sit, stand, walk - see ai/ava_action_met.py). Primary activity
# label AND primary live calorie source: each detected action has a hardcoded
# MET value, converted to kcal the same way the rest of the app does
# (kcal/min = MET * 3.5 * weight / 200). Falls back to pose_energy_estimator /
# calorie_regressor when no action clears YOWOv2's confidence threshold.
yowo_action_detector = YowoActionDetector()

def _smart_combine_activity(yowo_res, mc3_exercise, mc3_conf):
    STRONG_YOWO = {
        "run/jog": "Running",
        "swim": "Swimming",
        "ride (e.g., a bike, a car, a horse)": "Cycling",
        "dance": "HIIT",
        "martial art": "HIIT"
    }
    
    y_action = None
    y_conf = 0.0
    if yowo_res:
        y_action = yowo_res["action"]
        y_conf = yowo_res["confidence"]
        
    if y_action in STRONG_YOWO and y_conf > 0.3:
        final_activity = STRONG_YOWO[y_action]
    else:
        final_activity = mc3_exercise
        
    if y_action:
        # e.g. "Weightlifting - Sit"
        final_activity = f"{final_activity} - {y_action.title()}"
        
    return final_activity

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
class WorkoutFrameResponse(BaseModel):
    frame_number: int
    action_detected: str
    calories_burnt: float

    model_config = {"from_attributes": True}

class WorkoutHistoryResponse(BaseModel):
    id: int
    activity: str
    duration: float
    intensity: str
    calories: float
    created_at: Any

    model_config = {"from_attributes": True}

class WorkoutDetailResponse(WorkoutHistoryResponse):
    frames: Optional[List[WorkoutFrameResponse]] = None


@router.get("/history", response_model=List[WorkoutHistoryResponse])
def get_workout_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    workouts = db.query(Workout).filter(Workout.user_id == current_user.id).order_by(Workout.created_at.desc()).all()
    return workouts

@router.get("/history/{workout_id}", response_model=WorkoutDetailResponse)
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

        # 2. Activity recognition. Primary: YOWOv2 (pretrained, 80 AVA actions)
        # on a 16-frame sample spread across the clip. Fallback: MC3-18/UCF101
        # when YOWOv2 has no confident detection (e.g. no person clearly framed).
        activity, confidence = activity_recognizer.predict(frames)
        sample_idx = np.linspace(0, len(frames) - 1, min(16, len(frames))).astype(int)
        yowo_result = yowo_action_detector.predict_primary_action([frames[i] for i in sample_idx])
        
        activity = _smart_combine_activity(yowo_result, activity, confidence)
        if yowo_result:
            confidence = max(confidence, yowo_result["confidence"])
        
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
        
        # 5. Calorie estimation.
        #    Primary: end-to-end pose->energy regressor (uses the full pose
        #    sequence). Fallback: activity + movement_score -> XGBoost/MET when
        #    no trained pose->energy checkpoint is available.
        calories = None
        if pose_energy_estimator.available:
            calories = pose_energy_estimator.predict_calories(
                pose_history=pose_history,
                weight=u_weight,
                duration_seconds=workout_duration,
                fps=metadata["fps"],
            )
        if calories is None:
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
        current_activity = "Auto-Detect" # Default or client-defined
        frame_counter = 0
        workout_frames_data = []
        import asyncio
        yowo_state = {"is_detecting": False, "latest_activity": "Auto-Detect", "latest_met": None}
        
        # 2. Message Loop
        while True:
            data_str = await websocket.receive_text()
            message = json.loads(data_str)
            
            event = message.get("event")
            
            if event == "start_workout":
                current_activity = message.get("activity", "Auto-Detect")
                start_time = message.get("timestamp")
                pose_history = []
                cumulative_calories = 0.0
                last_frame_timestamp = None
                frame_counter = 0
                workout_frames_data = []
                yowo_state = {"is_detecting": False, "latest_activity": current_activity, "latest_met": None}
                await websocket.send_json({"status": "workout_started", "activity": current_activity})
                continue
                
            elif event == "frame_image":
                image_b64 = message.get("image")
                timestamp = message.get("timestamp")
                frame_counter += 1

                if not image_b64 or not start_time:
                    continue

                # Decode base64 JPEG → OpenCV BGR frame
                try:
                    if image_b64.startswith("data:"):
                        image_b64 = image_b64.split(",", 1)[1]
                    img_bytes = base64.b64decode(image_b64)
                    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        logger.warning("cv2.imdecode returned None")
                        continue

                    # Resize huge images to prevent event loop blocking
                    h, w = frame.shape[:2]
                    if max(h, w) > 640:
                        scale = 640.0 / max(h, w)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                except Exception:
                    continue

                # Auto-detect activity for this frame. On a 1 FPS live stream, a 16-frame
                # temporal buffer creates a disjointed montage that breaks YOWOv2.
                # To prevent 3D CNN inference from blocking the async websocket loop (which
                # causes ping timeouts and session drops), we run it in a background thread.
                if not yowo_state["is_detecting"]:
                    yowo_state["is_detecting"] = True
                    def run_yowo_inference(f, state):
                        try:
                            # Run BOTH models on the single frame
                            y_res = yowo_action_detector.predict_primary_action([f])
                            mc3_class, mc3_conf = activity_recognizer.predict([f])
                            
                            combined = _smart_combine_activity(y_res, mc3_class, mc3_conf)
                            state["latest_activity"] = combined
                            if y_res is not None:
                                state["latest_met"] = y_res["met"]
                        except Exception as e:
                            logger.error(f"Background AI error: {e}")
                        finally:
                            state["is_detecting"] = False
                            
                    asyncio.get_running_loop().run_in_executor(None, run_yowo_inference, frame, yowo_state)

                current_activity = yowo_state["latest_activity"]

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

                # Incremental accumulation. Primary: YOWOv2's detected action MET
                # (hardcoded per-action, see ava_action_met.py) -> kcal/sec, same
                # formula used everywhere else (MET * 3.5 * weight / 200 / 60).
                # Fallback 1: pose->energy regressor's own kcal/sec rate. Fallback
                # 2: MET-table/XGBoost formula, for when neither model is available.
                if last_frame_timestamp is not None:
                    time_slice = min(timestamp - last_frame_timestamp, 3.0)
                    if time_slice > 0:
                        calorie_rate = None
                        if yowo_state["latest_met"] is not None:
                            calorie_rate = yowo_state["latest_met"] * 3.5 * user.weight / 200.0 / 60.0
                        if calorie_rate is None and pose_energy_estimator.available:
                            calorie_rate = pose_energy_estimator.calorie_rate_per_second(
                                pose_history, user.weight
                            )
                        if calorie_rate is None:
                            calorie_rate = calorie_regressor.met_rate_per_second(
                                activity=current_activity,
                                weight=user.weight,
                                intensity=intensity_level,
                            )
                        frame_calories = calorie_rate * time_slice
                        cumulative_calories += frame_calories
                        
                        # Store frame data
                        workout_frames_data.append({
                            "frame_number": frame_counter,
                            "action_detected": current_activity,
                            "calories_burnt": frame_calories
                        })
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

                active_duration = max(0.1, timestamp - start_time)

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
                        calorie_rate = None
                        if pose_energy_estimator.available:
                            calorie_rate = pose_energy_estimator.calorie_rate_per_second(
                                pose_history, user.weight
                            )
                        if calorie_rate is None:
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
                    
                    final_activity = current_activity
                    if workout_frames_data:
                        from collections import Counter
                        activities = [f["action_detected"] for f in workout_frames_data]
                        final_activity = Counter(activities).most_common(1)[0][0]
                    
                    db_workout = Workout(
                        user_id=user.id,
                        activity=final_activity,
                        duration=duration,
                        intensity=final_intensity,
                        calories=cumulative_calories
                    )
                    db.add(db_workout)
                    db.commit()
                    db.refresh(db_workout)
                    
                    # Insert all stored frames
                    for frame_data in workout_frames_data:
                        db_frame = WorkoutFrame(
                            workout_id=db_workout.id,
                            frame_number=frame_data["frame_number"],
                            action_detected=frame_data["action_detected"],
                            calories_burnt=frame_data["calories_burnt"]
                        )
                        db.add(db_frame)
                    db.commit()
                    
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
        # Attempt to save the workout if the client dropped (e.g. phone sleep)
        if user and start_time and (last_frame_timestamp or 0) > start_time:
            duration = last_frame_timestamp - start_time
            if duration > 1.0:
                try:
                    final_activity = current_activity
                    if workout_frames_data:
                        from collections import Counter
                        activities = [f["action_detected"] for f in workout_frames_data]
                        final_activity = Counter(activities).most_common(1)[0][0]
                    
                    intensity_res = intensity_engine.calculate_intensity(pose_history, fps=2.0)
                    db_workout = Workout(
                        user_id=user.id,
                        activity=final_activity,
                        duration=duration,
                        intensity=intensity_res["intensity"],
                        calories=cumulative_calories
                    )
                    db.add(db_workout)
                    db.commit()
                    db.refresh(db_workout)
                    
                    for frame_data in workout_frames_data:
                        db_frame = WorkoutFrame(
                            workout_id=db_workout.id,
                            frame_number=frame_data["frame_number"],
                            action_detected=frame_data["action_detected"],
                            calories_burnt=frame_data["calories_burnt"]
                        )
                        db.add(db_frame)
                    db.commit()
                    logger.info(f"Auto-saved workout {db_workout.id} after disconnect.")
                except Exception as e:
                    logger.error(f"Failed to auto-save workout: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        db.close()
