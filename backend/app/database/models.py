from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from .connection import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    height = Column(Float, nullable=False)  # in cm
    weight = Column(Float, nullable=False)  # in kg
    gender = Column(String, nullable=False)  # 'M' or 'F'
    goal = Column(String, nullable=True)  # 'fat_to_fit', 'skinny_to_fit', or 'skinny_fat_to_fit'
    equipment_available = Column(Boolean, nullable=True)
    time_available = Column(String, nullable=True)  # '30_min', '1_hour', or '2_hour'

    workouts = relationship("Workout", back_populates="user", cascade="all, delete-orphan")

class Workout(Base):
    __tablename__ = "workouts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    activity = Column(String, nullable=False)
    duration = Column(Float, nullable=False)  # in seconds
    intensity = Column(String, nullable=False)  # 'Low', 'Medium', 'High'
    calories = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="workouts")
