from typing import Optional
from sqlalchemy.orm import Session
from db.models.user_model import UserModel, UserCreate
import hashlib


def get_user_by_email(db: Session, email: str) -> Optional[UserModel]:
    return db.query(UserModel).filter(UserModel.email == email).first()


def get_user_by_username(db: Session, username: str) -> Optional[UserModel]:
    return db.query(UserModel).filter(UserModel.username == username).first()


def create_user(db: Session, user: UserCreate, hashed_password: str) -> UserModel:
    db_user = UserModel(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        is_active=True,
        is_superuser=False
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hashed_password == hashlib.sha256(plain_password.encode()).hexdigest()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()