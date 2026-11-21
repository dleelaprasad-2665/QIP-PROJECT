"""Password hashing and session management."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from material_intelligence.app.config import get_settings
from material_intelligence.app.db.models import SessionToken, User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_session(db: Session, user: User) -> str:
    settings = get_settings()
    token = secrets.token_urlsafe(32)
    expires = datetime.now(UTC) + timedelta(seconds=settings.session_max_age_seconds)
    db.add(SessionToken(token=token, user_id=user.id, expires_at=expires))
    db.commit()
    return token


def revoke_session(db: Session, token: str) -> None:
    row = db.scalar(select(SessionToken).where(SessionToken.token == token))
    if row:
        db.delete(row)
        db.commit()


def get_user_by_session(db: Session, token: str | None) -> User | None:
    if not token:
        return None
    row = db.scalar(select(SessionToken).where(SessionToken.token == token))
    if row is None:
        return None
    expires = row.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    if expires < datetime.now(UTC):
        db.delete(row)
        db.commit()
        return None
    return row.user


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = db.scalar(select(User).where(User.email == email.lower()))
    if user is None or not user.is_active:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user
