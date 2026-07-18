from fastapi import APIRouter, Depends
from ..database.models import User
from .auth import get_current_user
from ..services.workout_plan import generate_plan
from ..services.meal_plan import generate_meal_plan

router = APIRouter(prefix="/api/plan", tags=["plan"])


@router.get("")
def get_workout_plan(current_user: User = Depends(get_current_user)):
    return generate_plan(
        user_id=current_user.id,
        goal=current_user.goal,
        equipment_available=current_user.equipment_available,
        time_available=current_user.time_available,
    )


@router.get("/meal")
def get_meal_plan(current_user: User = Depends(get_current_user)):
    return generate_meal_plan(
        user_id=current_user.id,
        age=current_user.age,
        height=current_user.height,
        weight=current_user.weight,
        gender=current_user.gender,
        goal=current_user.goal,
        food_preference=current_user.food_preference,
    )
