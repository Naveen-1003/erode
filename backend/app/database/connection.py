import os
import logging
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("burn_ex_database")

# Load database configuration from environment variables
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "burn_ex_db")

# Construct URL
POSTGRES_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
DATABASE_URL = os.getenv("DATABASE_URL", POSTGRES_URL)

engine = None
try:
    # Attempt to initialize PostgreSQL engine
    logger.info("Attempting connection to PostgreSQL...")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args={"connect_timeout": 5})
    # Run a quick check to see if database is reachable
    with engine.connect() as conn:
        logger.info("Successfully connected to PostgreSQL database!")
except Exception as e:
    logger.warning(
        f"PostgreSQL connection failed ({e}). "
        "Falling back to local SQLite database (burn_ex.db) for development."
    )
    DATABASE_URL = "sqlite:///./burn_ex.db"
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# (table, column, DDL type) added after the table already existed in earlier deployments.
_PENDING_COLUMNS = [
    ("users", "goal", "VARCHAR"),
    ("users", "equipment_available", "BOOLEAN"),
    ("users", "time_available", "VARCHAR"),
]

def run_pending_migrations():
    """Adds columns that were introduced after tables already existed.
    Base.metadata.create_all() only creates missing tables, not missing columns
    on tables that were created by an earlier version of the schema.
    """
    inspector = inspect(engine)
    for table, column, ddl_type in _PENDING_COLUMNS:
        if not inspector.has_table(table):
            continue
        existing_columns = {col["name"] for col in inspector.get_columns(table)}
        if column not in existing_columns:
            logger.info(f"Adding missing '{column}' column to {table} table.")
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
