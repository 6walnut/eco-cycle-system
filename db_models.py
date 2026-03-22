"""
Database models for macro datasets and analysis runs.
Default: SQLite file data/eco_cycle.db
Override: set env DATABASE_URL=mysql+pymysql://user:pass@host:3306/dbname
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class MacroDataset(Base):
    __tablename__ = "macro_datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), default="upload")
    # JSON array of row dicts: [{"date":"2010-01-01","cpi_yoy":1.5,...}, ...]
    rows_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    runs: Mapped[List["AnalysisRun"]] = relationship(back_populates="dataset")


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("macro_datasets.id"), index=True)
    # Request params as JSON
    params_json: Mapped[str] = mapped_column(Text)
    # Full API result JSON
    result_json: Mapped[str] = mapped_column(Text)
    forecast_model: Mapped[str] = mapped_column(String(32), default="hw")  # hw | lstm
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    dataset: Mapped["MacroDataset"] = relationship(back_populates="runs")


def get_database_url() -> str:
    env = os.environ.get("DATABASE_URL")
    if env:
        return env
    base = os.path.dirname(os.path.abspath(__file__))
    dbfile = os.path.join(base, "data", "eco_cycle.db")
    os.makedirs(os.path.dirname(dbfile), exist_ok=True)
    # SQLite URL: three slashes for relative path on Windows use absolute
    return f"sqlite:///{dbfile.replace(chr(92), '/')}"


def create_engine_from_env():
    url = get_database_url()
    if url.startswith("sqlite"):
        return create_engine(url, echo=False, future=True)
    return create_engine(url, echo=False, future=True, pool_pre_ping=True)


_engine = None
_SessionLocal = None


def init_db():
    global _engine, _SessionLocal
    _engine = create_engine_from_env()
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


def _session():
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


def save_dataset(rows: List[Dict[str, Any]], name: str = "upload") -> int:
    """Persist uploaded macro rows; returns dataset id."""
    init_db()
    with _session() as s:
        ds = MacroDataset(name=name, rows_json=json.dumps(rows, ensure_ascii=False))
        s.add(ds)
        s.commit()
        ds_id = ds.id
    return ds_id


def load_dataset_rows(dataset_id: int) -> List[Dict[str, Any]]:
    init_db()
    with _session() as s:
        ds = s.get(MacroDataset, dataset_id)
        if ds is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        return json.loads(ds.rows_json)


def save_analysis_run(
    dataset_id: int,
    params: Dict[str, Any],
    result: Dict[str, Any],
    forecast_model: str = "hw",
) -> int:
    init_db()
    with _session() as s:
        run = AnalysisRun(
            dataset_id=dataset_id,
            params_json=json.dumps(params, ensure_ascii=False),
            result_json=json.dumps(result, ensure_ascii=False),
            forecast_model=forecast_model,
        )
        s.add(run)
        s.commit()
        rid = run.id
    return rid


def list_datasets(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    with _session() as s:
        stmt = select(MacroDataset).order_by(MacroDataset.id.desc()).limit(limit)
        rows = s.scalars(stmt).all()
        out = []
        for d in rows:
            rlist = json.loads(d.rows_json)
            out.append(
                {
                    "id": d.id,
                    "name": d.name,
                    "rows": len(rlist),
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                }
            )
        return out


def get_analysis_run(run_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    with _session() as s:
        r = s.get(AnalysisRun, run_id)
        if r is None:
            return None
        return {
            "id": r.id,
            "dataset_id": r.dataset_id,
            "params": json.loads(r.params_json),
            "result": json.loads(r.result_json),
            "forecast_model": r.forecast_model,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
