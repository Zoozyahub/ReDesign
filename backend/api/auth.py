from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext

import jwt
from datetime import datetime, timedelta

from database.db import get_db  # подключаем функцию получения сессии
from database.db import User  # импорт модели

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Настройки для JWT токена
SECRET_KEY = "secret"  # Лучше использовать переменные окружения
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 часа

# ----------- Pydantic-схемы -----------

class RegisterRequest(BaseModel):
    name: str
    last_name: str
    email: EmailStr
    phone: str
    password: str
    role: int  # потом можно сделать Enum

class UserResponse(BaseModel):
    id: int
    name: str
    last_name: str
    email: str
    phone: str
    role: int

    class Config:
        orm_mode = True

class AuthRequest(BaseModel):
    email: EmailStr
    password: str

class AuthResponse(BaseModel):
    message: str
    token: str
    user: UserResponse


# ----------- Утилиты -----------

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# ----------- Эндпоинты -----------

@router.post("/register", response_model=AuthResponse)
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    # Проверка на существующего пользователя
    existing_user = db.query(User).filter(User.email == data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Хэширование пароля
    hashed_pwd = hash_password(data.password)

    new_user = User(
        name=data.name,
        last_name=data.last_name,
        email=data.email,
        phone=data.phone,
        password=hashed_pwd,
        role=data.role,
        created_at=datetime.utcnow()
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)  # Обновляем объект после коммита для получения id

    # Создаем токен доступа
    access_token = create_access_token(
        data={"sub": data.email, "user_id": new_user.id}
    )

    return {"message": "User registered successfully", "token": access_token, "user": new_user}

@router.post("/login", response_model=AuthResponse)
def login(data: AuthRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()

    if not user or not verify_password(data.password, user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    # Создаем токен доступа
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id}
    )

    return {"message": "Login successful", "token": access_token, "user": user}