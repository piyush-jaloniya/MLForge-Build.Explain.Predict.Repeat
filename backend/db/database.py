"""
backend/db/database.py — SQLAlchemy setup, models, session factory.
"""
from datetime import datetime
from typing import Generator
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from config import get_settings

settings = get_settings()
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── ORM Models ────────────────────────────────────────────────────────────────

class SessionRecord(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True)          # UUID
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    context_json = Column(Text, default="{}")       # Serialised ContextStore


class DatasetRecord(Base):
    __tablename__ = "datasets"
    id = Column(String, primary_key=True)           # UUID
    session_id = Column(String, index=True)
    filename = Column(String)
    file_path = Column(String)
    n_rows = Column(Integer)
    n_cols = Column(Integer)
    size_bytes = Column(Integer)
    column_types_json = Column(Text, default="{}")
    uploaded_at = Column(DateTime, default=datetime.utcnow)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"
    id = Column(String, primary_key=True)           # run_id UUID
    session_id = Column(String, index=True)
    model_name = Column(String)
    model_type = Column(String)                     # classical / dl / automl
    status = Column(String, default="queued")       # queued / running / complete / failed / cancelled
    progress = Column(Float, default=0.0)           # 0.0 – 1.0
    eta_seconds = Column(Float, nullable=True)
    metrics_json = Column(Text, default="{}")
    params_json = Column(Text, default="{}")
    feature_cols_json = Column(Text, default="[]")
    target_col = Column(String, nullable=True)
    task_type = Column(String, nullable=True)       # classification / regression
    artifact_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    mlflow_run_id = Column(String, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class ColumnAnnotation(Base):
    __tablename__ = "column_annotations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    dataset_id = Column(String)
    column_name = Column(String)
    annotation = Column(Text, default="")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PreprocessingStep(Base):
    __tablename__ = "preprocessing_steps"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    step_index = Column(Integer)
    step_type = Column(String)
    step_params_json = Column(Text, default="{}")
    applied_at = Column(DateTime, default=datetime.utcnow)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
