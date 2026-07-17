from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import auth, prediction, plan
from .database.connection import engine, Base, run_pending_migrations


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        Base.metadata.create_all(bind=engine)
        run_pending_migrations()
        print("Database tables verified/created on startup.")
    except Exception as e:
        print(f"Error during database initialization at startup: {e}")
    yield


app = FastAPI(
    title="Burn-Ex AI Fitness System API",
    description="Backend API for human action recognition, pose estimation, and calorie burn tracking.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for all origins to allow communication from mobile emulators and physical devices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router)
app.include_router(prediction.router)
app.include_router(plan.router)

# Root endpoint
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to the Burn-Ex AI Fitness Analytics API. Visit /docs for documentation."
    }
