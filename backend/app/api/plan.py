from fastapi import APIRouter, Depends
from ..database.models import User
from .auth import get_current_user
from ..services.workout_plan import generate_plan

router = APIRouter(prefix="/api/plan", tags=["plan"])


@router.get("")
def get_workout_plan(current_user: User = Depends(get_current_user)):
    return generate_plan(
        user_id=current_user.id,
        goal=current_user.goal,
        equipment_available=current_user.equipment_available,
        time_available=current_user.time_available,
    )
