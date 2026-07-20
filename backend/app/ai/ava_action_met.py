"""MET values for the 80 AVA v2.2 action classes used by YOWOv2.

AVA ships no calorie/MET labels - only action classes - so each value below is a
Compendium-of-Physical-Activities-informed estimate (same labeling strategy used
for the project's UTD-MHAD-trained model): APPROXIMATE, not a verbatim Compendium
code lookup. Sedentary/screen/talk-type actions sit near resting (~1.0-1.6 MET,
reflecting that AVA is sourced from movie/TV footage - mostly people sitting,
talking, watching), through light object/hand interactions (~1.8-2.5), to
moderate whole-body actions (~3-4.5), up to vigorous ones like running or
swimming (~6-8).

Keys match `config.dataset_config.dataset_config['ava_v2.2']['label_map']`
verbatim (including punctuation) so callers can index directly by model output.
"""

from typing import Dict, Optional

AVA_ACTION_MET: Dict[str, float] = {
    # Pose (1-14)
    "bend/bow(at the waist)": 3.0,
    "crawl": 5.0,
    "crouch/kneel": 2.8,
    "dance": 4.5,
    "fall down": 3.5,
    "get up": 3.0,
    "jump/leap": 6.0,
    "lie/sleep": 1.0,
    "martial art": 8.0,
    "run/jog": 7.0,
    "sit": 1.3,
    "stand": 1.8,
    "swim": 7.0,
    "walk": 3.0,
    # Person-object interaction (15-63)
    "answer phone": 1.5,
    "brush teeth": 2.0,
    "carry/hold (an object)": 2.5,
    "catch (an object)": 2.8,
    "chop": 2.5,
    "climb (e.g. a mountain)": 6.0,
    "clink glass": 1.6,
    "close (e.g., a door, a box)": 2.0,
    "cook": 2.5,
    "cut": 2.3,
    "dig": 4.0,
    "dress/put on clothing": 2.3,
    "drink": 1.5,
    "drive (e.g., a car, a truck)": 2.0,
    "eat": 1.5,
    "enter": 2.0,
    "exit": 2.0,
    "extract": 2.3,
    "fishing": 2.5,
    "hit (an object)": 4.0,
    "kick (an object)": 3.5,
    "lift/pick up": 3.0,
    "listen (e.g., to music)": 1.3,
    "open (e.g., a window, a car door)": 2.0,
    "paint": 2.3,
    "play board game": 1.5,
    "play musical instrument": 2.0,
    "play with pets": 2.8,
    "point to (an object)": 1.8,
    "press": 2.0,
    "pull (an object)": 3.5,
    "push (an object)": 3.5,
    "put down": 2.0,
    "read": 1.3,
    "ride (e.g., a bike, a car, a horse)": 4.0,
    "row boat": 4.5,
    "sail boat": 3.0,
    "shoot": 3.5,
    "shovel": 4.5,
    "smoke": 1.5,
    "stir": 2.0,
    "take a photo": 1.8,
    "text on/look at a cellphone": 1.4,
    "throw": 3.5,
    "touch (an object)": 1.8,
    "turn (e.g., a screwdriver)": 2.0,
    "watch (e.g., TV)": 1.3,
    "work on a computer": 1.5,
    "write": 1.5,
    # Person-person interaction (64-80)
    "fight/hit (a person)": 6.5,
    "give/serve (an object) to (a person)": 2.0,
    "grab (a person)": 3.0,
    "hand clap": 2.3,
    "hand shake": 1.8,
    "hand wave": 2.0,
    "hug (a person)": 1.8,
    "kick (a person)": 5.5,
    "kiss (a person)": 1.5,
    "lift (a person)": 4.5,
    "listen to (a person)": 1.4,
    "play with kids": 3.0,
    "push (another person)": 4.0,
    "sing to (e.g., self, a person, a group)": 1.8,
    "take (an object) from (a person)": 2.0,
    "talk to (e.g., self, a person, a group)": 1.6,
    "watch (a person)": 1.3,
}

_MET_MIN, _MET_MAX = 1.0, 8.0


def get_met(action_label: str, default: float = 2.0) -> float:
    """MET for an AVA action label, clamped to a physiologically sane range."""
    met = AVA_ACTION_MET.get(action_label, default)
    return max(_MET_MIN, min(_MET_MAX, met))


# AVA labels that map 1:1 onto one of the calorie model's 7-class vocabulary
# (see app/ai/calorie_model.py::ENCODER_CLASSES) with NO loss of specificity -
# "Running"/"Walking"/"Swimming"/"Cycling" are themselves exactly what a user
# would call that activity, not a generic catch-all. Deliberately does NOT
# include things like "martial art", "dance", "climb", "jump/leap", or
# "fight/hit (a person)": collapsing those to "HIIT" would throw away a
# perfectly specific, recognizable label for a vague bucket - exactly the
# problem this whole label-naming pass exists to avoid. Those instead get
# reported under their own raw label (see get_exercise_class returning None),
# or - for jump/leap specifically - deferred to MC3-18's more specific guess
# (e.g. "Jumping Jack") via prediction._POSE_AMBIGUOUS_ACTIONS.
AVA_TO_EXERCISE_CLASS: Dict[str, str] = {
    "run/jog": "Running",
    "walk": "Walking",
    "swim": "Swimming",
    "ride (e.g., a bike, a car, a horse)": "Cycling",
}


def get_exercise_class(action_label: str) -> Optional[str]:
    """The calorie model's exercise class for this AVA action, or None if it's
    an everyday (non-workout) action that should be reported as-is."""
    return AVA_TO_EXERCISE_CLASS.get(action_label)


# Curated "main" everyday actions worth reporting on their own. AVA has 80
# classes total, but most of the person-object/person-person ones (e.g.
# "clink glass", "shovel", "sail boat", "give/serve (an object) to (a
# person)") are narrow, visually similar to each other, and rarely what a
# fitness/activity tracker cares about - accepting any of them as the
# reported action trades accuracy for coverage nobody asked for. This list
# is deliberately short: the core body poses (sit/stand/walk/...) plus a
# handful of common, visually-distinctive social gestures. Edit freely to
# add/remove labels - every key must exist in AVA_ACTION_MET above.
MAIN_ACTIONS = {
    "sit",
    "stand",
    "walk",
    "bend/bow(at the waist)",
    "crouch/kneel",
    "lie/sleep",
    "get up",
    "fall down",
    "jump/leap",
    "run/jog",
    "dance",
    "martial art",
    "climb (e.g. a mountain)",
    "fight/hit (a person)",
    "hand wave",
    "hand shake",
    "hand clap",
    "hug (a person)",
}


def get_allowed_actions() -> "set[str]":
    """Every AVA label the app will ever report as a specific detected
    action: the curated everyday actions above, plus whatever maps to an
    exercise class (run/jog, swim, ride a bike, ...). Anything else AVA can
    output is intentionally never surfaced - see MAIN_ACTIONS' docstring."""
    return MAIN_ACTIONS | set(AVA_TO_EXERCISE_CLASS.keys())
