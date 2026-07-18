import os
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from ..database.connection import get_db
from ..database.models import User

# Configurations
SECRET_KEY = os.getenv("SECRET_KEY", "burn_ex_super_secret_key_change_me_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days for mobile sessions

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login-oauth")

router = APIRouter(prefix="/api/auth", tags=["auth"])

VALID_GOALS = {"fat_to_fit", "skinny_to_fit", "skinny_fat_to_fit"}
VALID_TIME_OPTIONS = {"30_min", "1_hour", "2_hour"}
VALID_FOOD_PREFERENCES = {"veg", "non_veg"}

# Pydantic Schemas
class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str
    age: int
    height: float  # in cm
    weight: float  # in kg
    gender: str    # 'M' or 'F'

class UserUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    gender: Optional[str] = None
    goal: Optional[str] = None
    equipment_available: Optional[bool] = None
    time_available: Optional[str] = None
    food_preference: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    age: int
    height: float
    weight: float
    gender: str
    goal: Optional[str] = None
    equipment_available: Optional[bool] = None
    time_available: Optional[str] = None
    food_preference: Optional[str] = None

    model_config = {"from_attributes": True}

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Get current user dependency
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

# API Routes
@router.post("/register", response_model=UserResponse)
def register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Standardize gender format
    gender_std = user_data.gender.upper().strip()
    if gender_std not in ["M", "F", "MALE", "FEMALE"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gender must be 'M' or 'F'"
        )
    gender_val = "M" if gender_std in ["M", "MALE"] else "F"

    hashed_pw = get_password_hash(user_data.password)
    db_user = User(
        name=user_data.name,
        email=user_data.email,
        password_hash=hashed_pw,
        age=user_data.age,
        height=user_data.height,
        weight=user_data.weight,
        gender=gender_val
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.post("/login", response_model=Token)
def login_user(login_data: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == login_data.email).first()
    if not user or not verify_password(login_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

# Standard OAuth2 form login for OpenAPI interactive docs
@router.post("/login-oauth")
def login_oauth(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.get("/me", response_model=UserResponse)
def get_user_me(current_user: User = Depends(get_current_user)):
    return current_user

@router.put("/profile", response_model=UserResponse)
def update_profile(
    profile_data: UserUpdate, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    if profile_data.name is not None:
        current_user.name = profile_data.name
    if profile_data.age is not None:
        current_user.age = profile_data.age
    if profile_data.height is not None:
        current_user.height = profile_data.height
    if profile_data.weight is not None:
        current_user.weight = profile_data.weight
    if profile_data.gender is not None:
        gender_std = profile_data.gender.upper().strip()
        if gender_std not in ["M", "F", "MALE", "FEMALE"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Gender must be 'M' or 'F'"
            )
        current_user.gender = "M" if gender_std in ["M", "MALE"] else "F"
    if profile_data.goal is not None:
        if profile_data.goal not in VALID_GOALS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Goal must be one of: {', '.join(sorted(VALID_GOALS))}"
            )
        current_user.goal = profile_data.goal
    if profile_data.equipment_available is not None:
        current_user.equipment_available = profile_data.equipment_available
    if profile_data.time_available is not None:
        if profile_data.time_available not in VALID_TIME_OPTIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Time available must be one of: {', '.join(sorted(VALID_TIME_OPTIONS))}"
            )
        current_user.time_available = profile_data.time_available
    if profile_data.food_preference is not None:
        if profile_data.food_preference not in VALID_FOOD_PREFERENCES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Food preference must be one of: {', '.join(sorted(VALID_FOOD_PREFERENCES))}"
            )
        current_user.food_preference = profile_data.food_preference

    db.commit()
    db.refresh(current_user)
    return current_user
