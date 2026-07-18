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
from ..ai.ava_action_met import get_exercise_class
from ..ai.kinetics_action_model import LifestyleActivityRecognizer, get_met as get_lifestyle_met
from ..ai.yoga_pose_classifier import classify_yoga_pose, get_met as get_yoga_met
from ..ai.walking_classifier import is_walking
from ..services.video_processing import VideoProcessor
from collections import Counter

logger = logging.getLogger("burn_ex_prediction")

router = APIRouter(prefix="/api/predict", tags=["prediction"])

# Frames kept in the live feed's rolling buffer, matching the 16-frame window
# YOWOv2/MC3-18/Kinetics-400 all expect. At the frontend's 1000ms capture
# interval this is ~16 real seconds of motion.
LIVE_ACTION_WINDOW = 16

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
# Broad everyday/lifestyle activity detector (washing dishes, laundry,
# generic yoga, ...) using torchvision's stock Kinetics-400 weights - see
# ai/kinetics_action_model.py. Neither YOWOv2/AVA nor MC3-18/UCF101 was ever
# trained to recognize these. Only actually invoked when neither of those two
# already has a confident, specific answer (see _should_check_lifestyle), so
# its extra inference cost is paid only when it can actually change the result.
lifestyle_recognizer = LifestyleActivityRecognizer()

# YOWOv2/AVA labels that are just a coarse, single-instant body pose - and so
# are visually indistinguishable, frame-to-frame, from the up/down or
# open/close cycle of a rep-based bodyweight exercise. A person doing squats
# alternates between poses AVA calls "stand" and "sit"/"crouch" every rep; a
# person doing jumping jacks alternates through "jump/leap"; AVA has no idea
# it's watching a repeated exercise rather than someone literally sitting
# down. MC3-18 looks at the whole 16-frame clip (not one instant) specifically
# to recognize that kind of repetition, so when YOWOv2 reports one of these
# AND MC3-18 confidently names one of its curated workouts, MC3-18's answer
# wins; failing that, Kinetics-400 gets a turn (e.g. someone standing at a
# sink washing dishes) - see _resolve_activity.
_POSE_AMBIGUOUS_ACTIONS = {"sit", "stand", "crouch/kneel", "bend/bow(at the waist)", "get up", "jump/leap"}


def _should_check_lifestyle(yowo_res, mc3_label) -> bool:
    """Whether it's worth running Kinetics-400 lifestyle detection at all:
    only when neither YOWOv2 nor MC3-18 already has a confident, specific
    answer - running a third heavy model when its result would never be used
    just burns latency for nothing. Mirrors the same condition _resolve_activity
    checks internally - keep both in sync if you change one."""
    if mc3_label != "Exercising":
        return False
    return yowo_res is None or yowo_res["action"] in _POSE_AMBIGUOUS_ACTIONS


def _resolve_activity(yoga_pose, yowo_res, mc3_label, mc3_bucket, kinetics_result, pose_window=None):
    """Return (display_activity, calorie_encoder_hint, met_hint), decided
    together so they can never disagree about which source's guess won.

    Priority:
    0. yoga_pose: a confident geometric match from the pose landmarks (see
       yoga_pose_classifier.py) - cheapest and most specific signal
       available; yoga poses are static/distinctive enough that a confident
       geometric match is trustworthy on its own.
    1. YOWOv2 "big cardio" mapping (run/jog, walk, swim, ride) - unambiguous
       and specific already.
    2. YOWOv2 ambiguous single-pose (_POSE_AMBIGUOUS_ACTIONS): prefer
       MC3-18's specific gym-workout guess if it has one; else Kinetics-400's
       specific lifestyle-activity guess if it has one; else the gait-based
       walking detector (walking_classifier.py) - catches a mid-stride
       walking pose that AVA misreads as "stand"/"sit"; else fall back to
       the raw YOWOv2 pose label.
    3. YOWOv2 other specific everyday action (wave, hug, handshake, clap,
       dance, martial art, climb, fight/hit, fall down, lie/sleep) - trust it.
    4. No YOWOv2 detection: prefer MC3-18, then Kinetics-400, then the
       gait-based walking detector, then a generic "Exercising".

    calorie_encoder_hint lets calorie_model.py skip fuzzy-matching the
    display text for the trained calorie model's required 7-class encoding
    (which can disagree with the source model's own bucket, e.g. "Push Ups"
    vs "Weightlifting"). met_hint is a precise, already-known MET (from
    YOWOv2's own AVA table, the yoga pose's hand-tuned MET, or Kinetics-400's)
    that callers can use directly for calorie math instead of re-deriving
    one. Both None when nothing more precise than the activity name itself is
    known (calorie_model.py's own AVA-aware matching applies instead).
    """
    if yoga_pose:
        return yoga_pose, None, get_yoga_met(yoga_pose)

    if yowo_res:
        y_action = yowo_res["action"]
        exercise = get_exercise_class(y_action)
        if exercise:
            return exercise, exercise, yowo_res["met"]
        if y_action in _POSE_AMBIGUOUS_ACTIONS:
            if mc3_label != "Exercising":
                return mc3_label, mc3_bucket, None
            if kinetics_result:
                return kinetics_result[0], None, get_lifestyle_met(kinetics_result[0])
            if pose_window and is_walking(pose_window):
                return "Walking", "Walking", None
        return y_action, None, yowo_res["met"]

    if mc3_label != "Exercising":
        return mc3_label, mc3_bucket, None
    if kinetics_result:
        return kinetics_result[0], None, get_lifestyle_met(kinetics_result[0])
    if pose_window and is_walking(pose_window):
        return "Walking", "Walking", None
    return "Exercising", "HIIT", None


def _majority_yoga_pose(pose_history: list) -> Optional[str]:
    """Most common geometric yoga-pose match across a clip's pose landmarks,
    requiring it to show up in a real share of frames (not just one noisy
    frame) before it's trusted as the clip's activity."""
    votes = [p for p in (classify_yoga_pose(lm) for lm in pose_history) if p]
    if not votes:
        return None
    label, count = Counter(votes).most_common(1)[0]
    return label if count / len(pose_history) >= 0.3 else None

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

        # 2. Pose landmark extraction (MediaPipe) - needed both for intensity
        # and for the geometric yoga-pose check below.
        pose_history = pose_estimator.process_video_frames(frames)

        # 3. Activity recognition: run all three action models, then let
        # _resolve_activity pick between them (MC3-18 wins for rep-based
        # exercises YOWOv2 can only see as an ambiguous single pose - e.g.
        # squats look like alternating "sit"/"stand" snapshots to YOWOv2 one
        # frame at a time; Kinetics-400 covers everyday/lifestyle activities
        # neither YOWOv2 nor MC3-18 was trained on, e.g. washing dishes).
        mc3_label, confidence, mc3_bucket = activity_recognizer.predict(frames)
        sample_idx = np.linspace(0, len(frames) - 1, min(16, len(frames))).astype(int)
        sampled_frames = [frames[i] for i in sample_idx]
        yowo_result = yowo_action_detector.predict_primary_action(sampled_frames)

        kinetics_result = None
        if lifestyle_recognizer.available and _should_check_lifestyle(yowo_result, mc3_label):
            kinetics_result = lifestyle_recognizer.predict(sampled_frames)

        yoga_pose = _majority_yoga_pose(pose_history)

        activity, calorie_hint, met_hint = _resolve_activity(
            yoga_pose, yowo_result, mc3_label, mc3_bucket, kinetics_result, pose_history
        )
        if yowo_result:
            confidence = max(confidence, yowo_result["confidence"])

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
                gender=u_gender,
                encoder_class_hint=calorie_hint,
                met_hint=met_hint,
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
        yowo_state = {"is_detecting": False, "latest_activity": "Auto-Detect", "latest_met": None, "latest_bucket_hint": None}
        # Rolling buffer of the most recent raw camera frames (capped at
        # ACTION_WINDOW, ~1 real frame/sec from the frontend's 1000ms capture
        # interval - see below). Motion-dependent activities (walking,
        # running, washing dishes, ...) are invisible to a single frozen
        # frame repeated 16 times, which is what this buffer replaces as the
        # input to the action models - see run_yowo_inference below.
        frame_buffer: List[np.ndarray] = []

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
                yowo_state = {"is_detecting": False, "latest_activity": current_activity, "latest_met": None, "latest_bucket_hint": None}
                frame_buffer = []
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

                frame_buffer.append(frame)
                if len(frame_buffer) > LIVE_ACTION_WINDOW:
                    frame_buffer.pop(0)

                # Auto-detect activity from the rolling frame_buffer (real motion
                # over the last ~LIVE_ACTION_WINDOW seconds at 1 FPS), not a single
                # frozen frame - motion-dependent activities (walking, washing
                # dishes, ...) are invisible to a repeated single frame, which is
                # what this app used until frame buffering was added. A short,
                # varied window can occasionally look like a "montage" rather than
                # one clean repetition, but that's a far smaller cost than having
                # zero motion signal at all.
                # To prevent 3D CNN inference from blocking the async websocket loop
                # (which causes ping timeouts and session drops), we run it in a
                # background thread on a snapshot of the buffer (copied so the main
                # loop can keep appending to frame_buffer while this runs).
                if not yowo_state["is_detecting"]:
                    yowo_state["is_detecting"] = True
                    def run_yowo_inference(frames_window, pose_window, state):
                        try:
                            # Run all three action models on the real frame window
                            y_res = yowo_action_detector.predict_primary_action(frames_window)
                            mc3_label, _, mc3_bucket = activity_recognizer.predict(frames_window)

                            kinetics_result = None
                            if lifestyle_recognizer.available and _should_check_lifestyle(y_res, mc3_label):
                                kinetics_result = lifestyle_recognizer.predict(frames_window)

                            state["latest_activity"], state["latest_bucket_hint"], state["latest_met"] = (
                                _resolve_activity(None, y_res, mc3_label, mc3_bucket, kinetics_result, pose_window)
                            )
                        except Exception as e:
                            logger.error(f"Background AI error: {e}")
                        finally:
                            state["is_detecting"] = False

                    asyncio.get_running_loop().run_in_executor(
                        None, run_yowo_inference, list(frame_buffer), list(pose_history), yowo_state
                    )

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

                # Cheap, synchronous geometric yoga-pose check (no model, just
                # joint-angle math on the landmarks we already have) - overrides
                # whatever the (slower, background-threaded) action models last
                # decided, since a confident geometric pose match is the most
                # specific signal available - see yoga_pose_classifier.py.
                yoga_pose = classify_yoga_pose(landmarks)
                yoga_met = get_yoga_met(yoga_pose) if yoga_pose else None
                if yoga_pose:
                    current_activity = yoga_pose

                pose_history.append(landmarks)
                if len(pose_history) > 30:  # 30 frames @ 1 FPS = 30-second window
                    pose_history.pop(0)

                # fps=1.0 matches the 1000ms capture interval on the frontend
                intensity_res = intensity_engine.calculate_intensity(pose_history, fps=1.0)
                movement_score = intensity_res["movement_score"]
                intensity_level = intensity_res["intensity"]

                # Incremental accumulation. Primary: the yoga pose's own MET, if
                # one was just detected. Fallback 1: YOWOv2's detected action MET
                # (hardcoded per-action, see ava_action_met.py) -> kcal/sec, same
                # formula used everywhere else (MET * 3.5 * weight / 200 / 60).
                # Fallback 2: pose->energy regressor's own kcal/sec rate. Fallback
                # 3: MET-table/XGBoost formula, for when neither model is available.
                if last_frame_timestamp is not None:
                    time_slice = min(timestamp - last_frame_timestamp, 3.0)
                    if time_slice > 0:
                        calorie_rate = None
                        if yoga_met is not None:
                            calorie_rate = yoga_met * 3.5 * user.weight / 200.0 / 60.0
                        if calorie_rate is None and yowo_state["latest_met"] is not None:
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
                                encoder_class_hint=yowo_state.get("latest_bucket_hint"),
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

                yoga_pose = classify_yoga_pose(landmarks)
                yoga_met = get_yoga_met(yoga_pose) if yoga_pose else None
                if yoga_pose:
                    current_activity = yoga_pose

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
                        if yoga_met is not None:
                            calorie_rate = yoga_met * 3.5 * user.weight / 200.0 / 60.0
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
