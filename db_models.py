"""
Database models for macro datasets, analysis runs and platform features.
Default: SQLite file data/eco_cycle.db
Override: set env DATABASE_URL=mysql+pymysql://user:pass@host:3306/dbname
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine, delete, select
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


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DatasetOwnership(Base):
    __tablename__ = "dataset_ownerships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("macro_datasets.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RunOwnership(Base):
    __tablename__ = "run_ownerships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_runs.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FavoriteRun(Base):
    __tablename__ = "favorite_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_runs.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ShareLink(Base):
    __tablename__ = "share_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    token: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_runs.id"), index=True)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SystemConfig(Base):
    __tablename__ = "system_configs"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value_json: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String(32), index=True)  # fusion | forecast
    name: Mapped[str] = mapped_column(String(64), index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    params_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(128), index=True)
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


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
    _seed_defaults()


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


def assign_dataset_owner(dataset_id: int, user_id: int) -> None:
    init_db()
    with _session() as s:
        row = DatasetOwnership(dataset_id=dataset_id, user_id=user_id)
        s.add(row)
        s.commit()


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
        used_ids = s.scalars(select(AnalysisRun.id).order_by(AnalysisRun.id.asc())).all()
        next_id = 1
        for rid in used_ids:
            if int(rid) != next_id:
                break
            next_id += 1
        run = AnalysisRun(
            id=next_id,
            dataset_id=dataset_id,
            params_json=json.dumps(params, ensure_ascii=False),
            result_json=json.dumps(result, ensure_ascii=False),
            forecast_model=forecast_model,
        )
        s.add(run)
        s.commit()
        rid = run.id
    return rid


def assign_run_owner(run_id: int, user_id: int) -> None:
    init_db()
    with _session() as s:
        row = RunOwnership(run_id=run_id, user_id=user_id)
        s.add(row)
        s.commit()


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


def create_user(username: str, password_hash: str, is_admin: bool = False) -> int:
    init_db()
    with _session() as s:
        exists = s.scalar(select(User).where(User.username == username))
        if exists is not None:
            raise ValueError("username already exists")
        u = User(username=username, password_hash=password_hash, is_admin=is_admin)
        s.add(u)
        s.commit()
        return u.id


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    init_db()
    with _session() as s:
        u = s.scalar(select(User).where(User.username == username))
        if u is None:
            return None
        return {
            "id": u.id,
            "username": u.username,
            "password_hash": u.password_hash,
            "is_admin": bool(u.is_admin),
            "created_at": u.created_at.isoformat() if u.created_at else None,
        }


def get_user(user_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    with _session() as s:
        u = s.get(User, user_id)
        if u is None:
            return None
        return {
            "id": u.id,
            "username": u.username,
            "is_admin": bool(u.is_admin),
            "created_at": u.created_at.isoformat() if u.created_at else None,
        }


def delete_user(user_id: int) -> bool:
    init_db()
    with _session() as s:
        u = s.get(User, user_id)
        if u is None:
            return False
        # Remove user-owned runs/datasets to keep "delete user data" consistent with admin behavior.
        run_owns = s.scalars(select(RunOwnership).where(RunOwnership.user_id == user_id)).all()
        ds_owns = s.scalars(select(DatasetOwnership).where(DatasetOwnership.user_id == user_id)).all()
        run_ids = [x.run_id for x in run_owns]
        ds_ids = [x.dataset_id for x in ds_owns]
        for rid in run_ids:
            s.execute(delete(FavoriteRun).where(FavoriteRun.run_id == rid))
            s.execute(delete(RunOwnership).where(RunOwnership.run_id == rid))
            s.execute(delete(AnalysisRun).where(AnalysisRun.id == rid))
        for did in ds_ids:
            s.execute(delete(DatasetOwnership).where(DatasetOwnership.dataset_id == did))
            s.execute(delete(MacroDataset).where(MacroDataset.id == did))
        s.execute(delete(FavoriteRun).where(FavoriteRun.user_id == user_id))
        s.execute(delete(RunOwnership).where(RunOwnership.user_id == user_id))
        s.execute(delete(DatasetOwnership).where(DatasetOwnership.user_id == user_id))
        s.execute(delete(ShareLink).where(ShareLink.created_by == user_id))
        s.delete(u)
        s.commit()
        return True


def list_users(limit: int = 200) -> List[Dict[str, Any]]:
    init_db()
    with _session() as s:
        rows = s.scalars(select(User).order_by(User.id.desc()).limit(limit)).all()
        return [
            {
                "id": u.id,
                "username": u.username,
                "is_admin": bool(u.is_admin),
                "created_at": u.created_at.isoformat() if u.created_at else None,
            }
            for u in rows
        ]


def list_user_datasets(user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    init_db()
    with _session() as s:
        own = s.scalars(
            select(DatasetOwnership).where(DatasetOwnership.user_id == user_id).order_by(DatasetOwnership.id.desc()).limit(limit)
        ).all()
        out: List[Dict[str, Any]] = []
        for o in own:
            ds = s.get(MacroDataset, o.dataset_id)
            if ds is None:
                continue
            rows = json.loads(ds.rows_json)
            out.append(
                {
                    "id": ds.id,
                    "name": ds.name,
                    "rows": len(rows),
                    "created_at": ds.created_at.isoformat() if ds.created_at else None,
                }
            )
        return out


def list_user_runs(user_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    init_db()
    with _session() as s:
        own = s.scalars(select(RunOwnership).where(RunOwnership.user_id == user_id).order_by(RunOwnership.id.desc()).limit(limit)).all()
        out: List[Dict[str, Any]] = []
        for o in own:
            r = s.get(AnalysisRun, o.run_id)
            if r is None:
                continue
            out.append(
                {
                    "id": r.id,
                    "dataset_id": r.dataset_id,
                    "forecast_model": r.forecast_model,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
            )
        return out


def user_owns_dataset(user_id: int, dataset_id: int) -> bool:
    init_db()
    with _session() as s:
        o = s.scalar(
            select(DatasetOwnership).where(DatasetOwnership.user_id == user_id, DatasetOwnership.dataset_id == dataset_id)
        )
        return o is not None


def user_owns_run(user_id: int, run_id: int) -> bool:
    init_db()
    with _session() as s:
        o = s.scalar(select(RunOwnership).where(RunOwnership.user_id == user_id, RunOwnership.run_id == run_id))
        return o is not None


def delete_analysis_run(run_id: int) -> bool:
    init_db()
    with _session() as s:
        r = s.get(AnalysisRun, run_id)
        if r is None:
            return False
        s.execute(delete(FavoriteRun).where(FavoriteRun.run_id == run_id))
        s.execute(delete(RunOwnership).where(RunOwnership.run_id == run_id))
        s.execute(delete(ShareLink).where(ShareLink.run_id == run_id))
        s.delete(r)
        s.commit()
        return True


def add_favorite_run(user_id: int, run_id: int) -> None:
    init_db()
    with _session() as s:
        ex = s.scalar(select(FavoriteRun).where(FavoriteRun.user_id == user_id, FavoriteRun.run_id == run_id))
        if ex is None:
            s.add(FavoriteRun(user_id=user_id, run_id=run_id))
            s.commit()


def remove_favorite_run(user_id: int, run_id: int) -> None:
    init_db()
    with _session() as s:
        s.execute(delete(FavoriteRun).where(FavoriteRun.user_id == user_id, FavoriteRun.run_id == run_id))
        s.commit()


def list_favorite_runs(user_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    init_db()
    with _session() as s:
        favs = s.scalars(select(FavoriteRun).where(FavoriteRun.user_id == user_id).order_by(FavoriteRun.id.desc()).limit(limit)).all()
        out = []
        for f in favs:
            r = s.get(AnalysisRun, f.run_id)
            if r is None:
                continue
            out.append(
                {
                    "run_id": r.id,
                    "dataset_id": r.dataset_id,
                    "forecast_model": r.forecast_model,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
            )
        return out


def create_share_link(token: str, run_id: int, created_by: int, expires_at: Optional[datetime]) -> int:
    init_db()
    with _session() as s:
        row = ShareLink(token=token, run_id=run_id, created_by=created_by, expires_at=expires_at)
        s.add(row)
        s.commit()
        return row.id


def get_share_link(token: str) -> Optional[Dict[str, Any]]:
    init_db()
    with _session() as s:
        row = s.scalar(select(ShareLink).where(ShareLink.token == token))
        if row is None:
            return None
        return {
            "id": row.id,
            "token": row.token,
            "run_id": row.run_id,
            "created_by": row.created_by,
            "expires_at": row.expires_at.isoformat() if row.expires_at else None,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }


def set_system_config(key: str, value: Any) -> None:
    init_db()
    with _session() as s:
        row = s.get(SystemConfig, key)
        payload = json.dumps(value, ensure_ascii=False)
        if row is None:
            s.add(SystemConfig(key=key, value_json=payload))
        else:
            row.value_json = payload
            row.updated_at = datetime.utcnow()
        s.commit()


def get_system_config(key: str, default: Any = None) -> Any:
    init_db()
    with _session() as s:
        row = s.get(SystemConfig, key)
        if row is None:
            return default
        try:
            return json.loads(row.value_json)
        except Exception:
            return default


def list_system_configs() -> Dict[str, Any]:
    init_db()
    with _session() as s:
        rows = s.scalars(select(SystemConfig)).all()
        out: Dict[str, Any] = {}
        for r in rows:
            try:
                out[r.key] = json.loads(r.value_json)
            except Exception:
                out[r.key] = r.value_json
        return out


def upsert_model(model_type: str, name: str, enabled: bool, params: Dict[str, Any]) -> int:
    init_db()
    with _session() as s:
        row = s.scalar(select(ModelRegistry).where(ModelRegistry.model_type == model_type, ModelRegistry.name == name))
        payload = json.dumps(params or {}, ensure_ascii=False)
        if row is None:
            row = ModelRegistry(model_type=model_type, name=name, enabled=enabled, params_json=payload)
            s.add(row)
            s.commit()
            return row.id
        row.enabled = enabled
        row.params_json = payload
        s.commit()
        return row.id


def delete_model(model_id: int) -> bool:
    init_db()
    with _session() as s:
        row = s.get(ModelRegistry, model_id)
        if row is None:
            return False
        s.delete(row)
        s.commit()
        return True


def list_models(model_type: Optional[str] = None) -> List[Dict[str, Any]]:
    init_db()
    with _session() as s:
        stmt = select(ModelRegistry).order_by(ModelRegistry.id.asc())
        if model_type:
            stmt = stmt.where(ModelRegistry.model_type == model_type)
        rows = s.scalars(stmt).all()
        out = []
        for r in rows:
            try:
                params = json.loads(r.params_json)
            except Exception:
                params = {}
            out.append(
                {
                    "id": r.id,
                    "model_type": r.model_type,
                    "name": r.name,
                    "enabled": bool(r.enabled),
                    "params": params,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
            )
        return out


def add_audit_log(action: str, user_id: Optional[int] = None, username: Optional[str] = None, detail: Optional[str] = None) -> int:
    init_db()
    with _session() as s:
        row = AuditLog(action=action, user_id=user_id, username=username, detail=detail)
        s.add(row)
        s.commit()
        return row.id


def list_audit_logs(limit: int = 200) -> List[Dict[str, Any]]:
    init_db()
    with _session() as s:
        rows = s.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)).all()
        out = []
        for r in rows:
            dt = r.created_at or datetime.utcnow()
            out.append(
                {
                    "id": r.id,
                    "date": dt.strftime("%Y-%m-%d"),
                    "time": dt.strftime("%H:%M:%S"),
                    "user_id": r.user_id,
                    "username": r.username,
                    "action": r.action,
                    "detail": r.detail,
                }
            )
        return out


def get_user_run_stats() -> List[Dict[str, Any]]:
    init_db()
    with _session() as s:
        users = s.scalars(select(User).order_by(User.id.asc())).all()
        out: List[Dict[str, Any]] = []
        for u in users:
            run_count = s.query(RunOwnership).filter(RunOwnership.user_id == u.id).count()
            out.append({"user_id": u.id, "username": u.username, "run_count": int(run_count)})
        return out


def _seed_defaults() -> None:
    with _session() as s:
        # default admin account
        if s.scalar(select(User).where(User.username == "admin")) is None:
            # Lazy import to avoid hard dependency at module import time
            from werkzeug.security import generate_password_hash

            s.add(User(username="admin", password_hash=generate_password_hash("admin123"), is_admin=True))
        # seed demo users requested by product requirements
        seed_users = [("fjy", "123456"), ("ria", "zxcvbn")]
        for uname, pwd in seed_users:
            if s.scalar(select(User).where(User.username == uname)) is None:
                from werkzeug.security import generate_password_hash

                s.add(User(username=uname, password_hash=generate_password_hash(pwd), is_admin=False))

        # default configs
        default_configs = {
            "akshare_indicators": {
                "cpi_yoy": True,
                "pmi": True,
                "m2_yoy": True,
                "m1_yoy": True,
                "social_finance_yoy": False,
            },
            "akshare_update_interval_minutes": 60,
        }
        for k, v in default_configs.items():
            if s.get(SystemConfig, k) is None:
                s.add(SystemConfig(key=k, value_json=json.dumps(v, ensure_ascii=False)))

        # default model registry
        defaults = [
            ("fusion", "equal", True, {}),
            ("fusion", "pca", True, {}),
            ("fusion", "entropy", True, {}),
            ("fusion", "dfm", True, {}),
            ("forecast", "hw", True, {}),
            ("forecast", "lstm", True, {"lookback": 12}),
        ]
        for model_type, name, enabled, params in defaults:
            ex = s.scalar(select(ModelRegistry).where(ModelRegistry.model_type == model_type, ModelRegistry.name == name))
            if ex is None:
                s.add(
                    ModelRegistry(
                        model_type=model_type,
                        name=name,
                        enabled=enabled,
                        params_json=json.dumps(params, ensure_ascii=False),
                    )
                )
        s.commit()
