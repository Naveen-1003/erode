import os
import sys
import logging

# Add current path to sys.path so we can import from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from app.database.connection import engine, Base
from app.database import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("burn_ex_setup")

def init_db():
    logger.info("Initializing database tables...")
    try:
        # Create all tables defined in models
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing database tables: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_db()
