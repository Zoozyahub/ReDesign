from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import base64
import os
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

# Импорт моделей и зависимостей
from database.db import get_db, Design, User

router = APIRouter()

# ----------- Pydantic-модели -----------

class ImageResponse(BaseModel):
    id: int
    author: str  # Будем использовать имя автора
    base64_image: str  # Изображение в формате base64

# ----------- Утилиты -----------

def image_to_base64(image_path: str) -> str:
    """Конвертирует изображение в base64-строку"""
    if not os.path.exists(image_path):
        # Если файл не найден, возвращаем placeholder
        return ""
    
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# ----------- Эндпоинты -----------

@router.get("/images", response_model=List[ImageResponse])
def get_all_images(db: Session = Depends(get_db)):
    """Получение всех публичных дизайнов"""
    designs = db.query(Design).filter(Design.is_public == True).all()
    
    result = []
    for design in designs:
        # Получаем имя и фамилию автора
        author_name = f"{design.user.name} {design.user.last_name}"
        
        # Получаем изображение как base64
        base64_image = image_to_base64(design.result_image_path)
        
        result.append({
            "id": design.id,
            "author": author_name,
            "base64_image": base64_image
        })
    
    return result

@router.get("/images/user/{user_id}", response_model=List[ImageResponse])
def get_user_images(user_id: int, db: Session = Depends(get_db)):
    """Получение всех дизайнов конкретного пользователя"""
    # Проверяем, существует ли пользователь
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Пользователь с ID {user_id} не найден"
        )
    
    # Получаем все дизайны пользователя
    designs = db.query(Design).filter(Design.user_id == user_id).all()
    
    result = []
    for design in designs:
        # Получаем имя и фамилию автора
        author_name = f"{user.name} {user.last_name}"
        
        # Получаем изображение как base64
        base64_image = image_to_base64(design.result_image_path)
        
        result.append({
            "id": design.id,
            "author": author_name,
            "base64_image": base64_image
        })
    
    return result