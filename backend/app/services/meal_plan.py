import random
from datetime import date
from typing import Any, Dict, List, Optional

BREAKFAST: List[Dict[str, Any]] = [
    {"name": "Oats with Banana & Peanut Butter", "calories": 350, "protein_g": 12, "carbs_g": 52, "fat_g": 11, "diet": "veg"},
    {"name": "Vegetable Poha", "calories": 300, "protein_g": 7, "carbs_g": 55, "fat_g": 7, "diet": "veg"},
    {"name": "Greek Yogurt with Berries & Granola", "calories": 320, "protein_g": 18, "carbs_g": 42, "fat_g": 9, "diet": "veg"},
    {"name": "Paneer Bhurji with Whole Wheat Toast", "calories": 400, "protein_g": 20, "carbs_g": 38, "fat_g": 18, "diet": "veg"},
    {"name": "Moong Dal Chilla", "calories": 310, "protein_g": 16, "carbs_g": 40, "fat_g": 9, "diet": "veg"},
    {"name": "Masala Omelette (2 Eggs) with Toast", "calories": 380, "protein_g": 22, "carbs_g": 30, "fat_g": 18, "diet": "non_veg"},
    {"name": "Boiled Eggs with Avocado Toast", "calories": 420, "protein_g": 20, "carbs_g": 34, "fat_g": 22, "diet": "non_veg"},
    {"name": "Chicken Sausage & Veggie Scramble", "calories": 400, "protein_g": 28, "carbs_g": 18, "fat_g": 24, "diet": "non_veg"},
]

LUNCH: List[Dict[str, Any]] = [
    {"name": "Rajma Chawal", "calories": 550, "protein_g": 20, "carbs_g": 90, "fat_g": 10, "diet": "veg"},
    {"name": "Paneer Tikka with Quinoa", "calories": 600, "protein_g": 28, "carbs_g": 60, "fat_g": 24, "diet": "veg"},
    {"name": "Chole with Brown Rice", "calories": 580, "protein_g": 20, "carbs_g": 92, "fat_g": 12, "diet": "veg"},
    {"name": "Dal Tadka with Roti & Salad", "calories": 500, "protein_g": 18, "carbs_g": 75, "fat_g": 12, "diet": "veg"},
    {"name": "Vegetable Pulao with Raita", "calories": 520, "protein_g": 14, "carbs_g": 80, "fat_g": 14, "diet": "veg"},
    {"name": "Grilled Chicken Breast with Brown Rice & Veggies", "calories": 620, "protein_g": 45, "carbs_g": 65, "fat_g": 14, "diet": "non_veg"},
    {"name": "Chicken Curry with Roti", "calories": 600, "protein_g": 38, "carbs_g": 58, "fat_g": 20, "diet": "non_veg"},
    {"name": "Fish Curry with Rice", "calories": 560, "protein_g": 34, "carbs_g": 62, "fat_g": 16, "diet": "non_veg"},
]

SNACK: List[Dict[str, Any]] = [
    {"name": "Roasted Chickpeas", "calories": 180, "protein_g": 8, "carbs_g": 25, "fat_g": 5, "diet": "veg"},
    {"name": "Mixed Nuts & Seeds", "calories": 220, "protein_g": 7, "carbs_g": 10, "fat_g": 18, "diet": "veg"},
    {"name": "Fruit Chaat", "calories": 150, "protein_g": 2, "carbs_g": 36, "fat_g": 1, "diet": "veg"},
    {"name": "Protein Shake", "calories": 200, "protein_g": 24, "carbs_g": 12, "fat_g": 5, "diet": "veg"},
    {"name": "Sprouts Salad", "calories": 170, "protein_g": 10, "carbs_g": 24, "fat_g": 4, "diet": "veg"},
    {"name": "Greek Yogurt Cup", "calories": 160, "protein_g": 14, "carbs_g": 16, "fat_g": 4, "diet": "veg"},
    {"name": "Boiled Eggs (2)", "calories": 180, "protein_g": 14, "carbs_g": 2, "fat_g": 12, "diet": "non_veg"},
    {"name": "Grilled Chicken Strips", "calories": 210, "protein_g": 26, "carbs_g": 4, "fat_g": 9, "diet": "non_veg"},
]

DINNER: List[Dict[str, Any]] = [
    {"name": "Grilled Paneer with Sauteed Vegetables", "calories": 520, "protein_g": 26, "carbs_g": 30, "fat_g": 30, "diet": "veg"},
    {"name": "Dal + Roti + Sabzi", "calories": 500, "protein_g": 18, "carbs_g": 78, "fat_g": 12, "diet": "veg"},
    {"name": "Vegetable Stir-Fry with Tofu", "calories": 480, "protein_g": 22, "carbs_g": 48, "fat_g": 20, "diet": "veg"},
    {"name": "Khichdi with Curd", "calories": 460, "protein_g": 16, "carbs_g": 70, "fat_g": 12, "diet": "veg"},
    {"name": "Grilled Chicken with Steamed Vegetables", "calories": 550, "protein_g": 42, "carbs_g": 30, "fat_g": 24, "diet": "non_veg"},
    {"name": "Baked Fish with Salad", "calories": 480, "protein_g": 38, "carbs_g": 20, "fat_g": 22, "diet": "non_veg"},
    {"name": "Egg Bhurji with Roti", "calories": 500, "protein_g": 24, "carbs_g": 50, "fat_g": 20, "diet": "non_veg"},
]

MEAL_POOLS: Dict[str, List[Dict[str, Any]]] = {
    "breakfast": BREAKFAST,
    "lunch": LUNCH,
    "snack": SNACK,
    "dinner": DINNER,
}

# Fraction of daily target calories allotted to each meal slot.
MEAL_CALORIE_SPLIT: Dict[str, float] = {
    "breakfast": 0.25,
    "lunch": 0.30,
    "snack": 0.15,
    "dinner": 0.30,
}

# Flat "lightly active" multiplier (Mifflin-St Jeor 1-3 days/week tier). There's no
# weekly training-frequency field on the user, so this is applied uniformly rather
# than derived from time_available/equipment_available (which describe a single
# session's shape, not how often the user trains).
ACTIVITY_MULTIPLIER = 1.375

# Calorie target as a multiplier of TDEE, per goal.
GOAL_CALORIE_ADJUSTMENT: Dict[str, float] = {
    "fat_to_fit": 0.80,
    "skinny_to_fit": 1.15,
    "skinny_fat_to_fit": 0.95,
}

# (protein %, carbs %, fat %) of daily calories, per goal.
GOAL_MACRO_SPLIT: Dict[str, Dict[str, float]] = {
    "fat_to_fit": {"protein": 0.35, "carbs": 0.35, "fat": 0.30},
    "skinny_to_fit": {"protein": 0.25, "carbs": 0.50, "fat": 0.25},
    "skinny_fat_to_fit": {"protein": 0.30, "carbs": 0.40, "fat": 0.30},
}

MIN_TARGET_CALORIES = 1200
MAX_TARGET_CALORIES = 4000

DEFAULT_GOAL = "skinny_fat_to_fit"
DEFAULT_FOOD_PREFERENCE = "veg"


def _bmr(age: int, height: float, weight: float, gender: str) -> float:
    """Mifflin-St Jeor basal metabolic rate."""
    base = 10 * weight + 6.25 * height - 5 * age
    return base + 5 if gender == "M" else base - 161


def _round_portion_multiplier(raw_multiplier: float) -> float:
    clamped = max(0.5, min(2.0, raw_multiplier))
    return round(clamped * 4) / 4


def generate_meal_plan(
    user_id: int,
    age: int,
    height: float,
    weight: float,
    gender: str,
    goal: Optional[str],
    food_preference: Optional[str],
) -> Dict[str, Any]:
    """Rule-based meal plan for today, scaled by goal/body stats/food preference.
    Seeded by user+date so the plan is stable through the day and rotates daily."""
    resolved_goal = goal if goal in GOAL_CALORIE_ADJUSTMENT else DEFAULT_GOAL
    resolved_food_preference = food_preference if food_preference in ("veg", "non_veg") else DEFAULT_FOOD_PREFERENCE

    rng = random.Random(f"{user_id}-{date.today().isoformat()}-{resolved_goal}-{resolved_food_preference}")

    tdee = _bmr(age, height, weight, gender) * ACTIVITY_MULTIPLIER
    target_calories = tdee * GOAL_CALORIE_ADJUSTMENT[resolved_goal]
    target_calories = max(MIN_TARGET_CALORIES, min(MAX_TARGET_CALORIES, target_calories))

    macro_split = GOAL_MACRO_SPLIT[resolved_goal]
    target_protein_g = target_calories * macro_split["protein"] / 4
    target_carbs_g = target_calories * macro_split["carbs"] / 4
    target_fat_g = target_calories * macro_split["fat"] / 9

    meals: Dict[str, Dict[str, Any]] = {}
    totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}

    for slot, pool in MEAL_POOLS.items():
        eligible = [item for item in pool if item["diet"] == "veg" or resolved_food_preference == "non_veg"]
        item = rng.choice(eligible)

        target_slot_calories = target_calories * MEAL_CALORIE_SPLIT[slot]
        multiplier = _round_portion_multiplier(target_slot_calories / item["calories"])

        meal = {
            "name": item["name"],
            "calories": round(item["calories"] * multiplier),
            "protein_g": round(item["protein_g"] * multiplier),
            "carbs_g": round(item["carbs_g"] * multiplier),
            "fat_g": round(item["fat_g"] * multiplier),
            "portion_label": f"{multiplier}x serving",
        }
        meals[slot] = meal
        totals["calories"] += meal["calories"]
        totals["protein_g"] += meal["protein_g"]
        totals["carbs_g"] += meal["carbs_g"]
        totals["fat_g"] += meal["fat_g"]

    return {
        "goal": resolved_goal,
        "food_preference": resolved_food_preference,
        "target_calories": round(target_calories),
        "target_protein_g": round(target_protein_g),
        "target_carbs_g": round(target_carbs_g),
        "target_fat_g": round(target_fat_g),
        "meals": meals,
        "totals": {k: round(v) for k, v in totals.items()},
    }
