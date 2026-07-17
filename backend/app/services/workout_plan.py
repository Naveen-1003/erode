import random
from datetime import date
from typing import Any, Dict, List, Optional

WARMUP: List[Dict[str, Any]] = [
    {"name": "Jumping Jacks", "duration_sec": 60},
    {"name": "Arm Circles", "duration_sec": 30},
    {"name": "High Knees", "duration_sec": 45},
    {"name": "Torso Twists", "duration_sec": 30},
    {"name": "Leg Swings", "duration_sec": 30},
]

COOLDOWN: List[Dict[str, Any]] = [
    {"name": "Standing Quad Stretch", "duration_sec": 30},
    {"name": "Seated Forward Fold", "duration_sec": 30},
    {"name": "Child's Pose", "duration_sec": 45},
    {"name": "Deep Breathing", "duration_sec": 60},
    {"name": "Shoulder Stretch", "duration_sec": 30},
]

BODYWEIGHT_STRENGTH: List[Dict[str, Any]] = [
    {"name": "Push-Ups", "sets": 3, "reps": 12},
    {"name": "Bodyweight Squats", "sets": 3, "reps": 15},
    {"name": "Walking Lunges", "sets": 3, "reps": 12},
    {"name": "Plank", "sets": 3, "duration_sec": 45},
    {"name": "Glute Bridges", "sets": 3, "reps": 15},
    {"name": "Tricep Dips", "sets": 3, "reps": 10},
    {"name": "Wall Push-Ups", "sets": 3, "reps": 15},
    {"name": "Superman Hold", "sets": 3, "duration_sec": 30},
]

EQUIPMENT_STRENGTH: List[Dict[str, Any]] = [
    {"name": "Dumbbell Bench Press", "sets": 4, "reps": 10},
    {"name": "Barbell Squats", "sets": 4, "reps": 10},
    {"name": "Dumbbell Rows", "sets": 3, "reps": 12},
    {"name": "Kettlebell Swings", "sets": 3, "reps": 15},
    {"name": "Dumbbell Deadlifts", "sets": 4, "reps": 8},
    {"name": "Resistance Band Rows", "sets": 3, "reps": 15},
    {"name": "Dumbbell Shoulder Press", "sets": 3, "reps": 10},
    {"name": "Cable/Band Bicep Curls", "sets": 3, "reps": 12},
]

CARDIO_HIIT: List[Dict[str, Any]] = [
    {"name": "Burpees", "sets": 3, "reps": 12},
    {"name": "Jump Rope", "sets": 3, "duration_sec": 60},
    {"name": "Squat Jumps", "sets": 3, "reps": 15},
    {"name": "High Knees", "sets": 3, "duration_sec": 40},
    {"name": "Butt Kicks", "sets": 3, "duration_sec": 40},
    {"name": "Mountain Climbers", "sets": 3, "duration_sec": 30},
    {"name": "Jumping Jacks", "sets": 3, "duration_sec": 45},
]

# (main exercise count, warmup count, cooldown count, estimated total minutes)
TIME_PROFILES: Dict[str, Dict[str, int]] = {
    "30_min": {"main_count": 4, "warmup_count": 1, "cooldown_count": 1, "estimated_minutes": 30},
    "1_hour": {"main_count": 6, "warmup_count": 2, "cooldown_count": 2, "estimated_minutes": 60},
    "2_hour": {"main_count": 9, "warmup_count": 3, "cooldown_count": 2, "estimated_minutes": 120},
}

# Fraction of the main exercise slots pulled from the cardio/HIIT pool vs. the strength pool.
GOAL_CARDIO_RATIO: Dict[str, float] = {
    "fat_to_fit": 0.65,
    "skinny_to_fit": 0.15,
    "skinny_fat_to_fit": 0.4,
}

GOAL_FOCUS_LABEL: Dict[str, str] = {
    "fat_to_fit": "Calorie Burn & Conditioning",
    "skinny_to_fit": "Strength & Muscle Building",
    "skinny_fat_to_fit": "Balanced Strength + Cardio",
}

DEFAULT_GOAL = "skinny_fat_to_fit"
DEFAULT_TIME = "1_hour"


def generate_plan(
    user_id: int,
    goal: Optional[str],
    equipment_available: Optional[bool],
    time_available: Optional[str],
) -> Dict[str, Any]:
    """Rule-based workout plan for today, scaled by goal/equipment/time.
    Seeded by user+date so the plan is stable through the day and rotates daily."""
    resolved_goal = goal if goal in GOAL_CARDIO_RATIO else DEFAULT_GOAL
    resolved_time = time_available if time_available in TIME_PROFILES else DEFAULT_TIME
    profile = TIME_PROFILES[resolved_time]

    rng = random.Random(f"{user_id}-{date.today().isoformat()}-{resolved_goal}-{equipment_available}-{resolved_time}")

    strength_pool = EQUIPMENT_STRENGTH + BODYWEIGHT_STRENGTH if equipment_available else BODYWEIGHT_STRENGTH
    cardio_count = round(profile["main_count"] * GOAL_CARDIO_RATIO[resolved_goal])
    strength_count = profile["main_count"] - cardio_count

    main_exercises = (
        rng.sample(CARDIO_HIIT, min(cardio_count, len(CARDIO_HIIT)))
        + rng.sample(strength_pool, min(strength_count, len(strength_pool)))
    )
    rng.shuffle(main_exercises)

    return {
        "goal": resolved_goal,
        "focus": GOAL_FOCUS_LABEL[resolved_goal],
        "equipment_available": bool(equipment_available),
        "time_available": resolved_time,
        "estimated_minutes": profile["estimated_minutes"],
        "warmup": rng.sample(WARMUP, min(profile["warmup_count"], len(WARMUP))),
        "exercises": main_exercises,
        "cooldown": rng.sample(COOLDOWN, min(profile["cooldown_count"], len(COOLDOWN))),
    }
