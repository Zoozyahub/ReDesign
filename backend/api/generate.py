from fastapi import FastAPI, UploadFile, Form, File, HTTPException, BackgroundTasks, APIRouter, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime
import json
import base64
from io import BytesIO
from PIL import Image
from sqlalchemy.orm import Session
from decimal import Decimal

# Import your existing modules
from models.generation import InteriorGenerator
from models.segmentation import FurnitureSegmenter
from models.search import FurnitureVectorSearch

from database.db import get_db  # подключаем функцию получения сессии
from database.db import User, Furniture, Design, DesignFurniture

router = APIRouter()

# Initialize your pipeline components
generator = InteriorGenerator()
segmenter = FurnitureSegmenter(model_type="coco")
searcher = FurnitureVectorSearch(db_path="./chroma_db", collection_name="furniture_clip")

# Create directories for storing uploaded and generated images
UPLOAD_DIR = Path("./uploads")
GENERATED_DIR = Path("./generated_interiors")
SEGMENTED_DIR = Path("./segmented_furniture")

UPLOAD_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(exist_ok=True)
SEGMENTED_DIR.mkdir(exist_ok=True)

# Model for furniture item response
class FurnitureItem(BaseModel):
    id: int
    name: str
    description: str
    price: str
    image: str
    url: str

# Model for the complete response
class GenerationResponse(BaseModel):
    generatedImage: str
    recommendedFurniture: List[FurnitureItem]
    designId: int

# Clean up temporary files
def cleanup_temp_files(file_paths: List[str]):
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Error cleaning up file {path}: {e}")

@router.post("/generate", response_model=GenerationResponse)
async def generate_interior(
    background_tasks: BackgroundTasks,
    userId: str = Form(...), 
    image: UploadFile = File(...),
    mode: str = Form(...),
    is_public: bool = Form(...),
    roomType: Optional[str] = Form(None),
    interiorStyle: Optional[str] = Form(None),
    textPrompt: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    # Create unique directories for this user's request
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    request_id = f"{userId}_{timestamp}_{uuid.uuid4().hex[:8]}"
    
    user_upload_dir = UPLOAD_DIR / userId
    user_upload_dir.mkdir(exist_ok=True)
    
    user_generated_dir = GENERATED_DIR / userId
    user_generated_dir.mkdir(exist_ok=True)
    
    # Files to clean up after response is sent
    temp_files = []
    
    try:
        # Save the uploaded image
        input_image_path = user_upload_dir / f"input_{request_id}.jpg"
        with open(input_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Create prompt based on the mode
        if mode == "quick":
            if not roomType or not interiorStyle:
                raise HTTPException(status_code=400, detail="Room type and interior style are required for quick mode")
            user_prompt = f"{roomType}, {interiorStyle} style"
        else:  # text mode
            if not textPrompt:
                raise HTTPException(status_code=400, detail="Text prompt is required for text mode")
            user_prompt = textPrompt
        
        # Generate the interior design
        results = generator.generate(
            input_image_path=str(input_image_path),
            user_prompt=user_prompt,
            output_dir=str(user_generated_dir),
            seed=42,
            return_image=True,
        )
        
        # Get the path to the enhanced image (or the original if enhancement wasn't applied)
        generated_image_path = results.get("enhanced_path") if "enhanced_path" in results else results.get("original_path")
        
        # Segment the furniture in the generated image
        results_seg = segmenter.segment_image(generated_image_path)
        
        # Find similar furniture items
        search_results = searcher.find_similar_furniture(results_seg)
        
        # Save relative paths for database storage
        rel_input_path = os.path.relpath(input_image_path, start=os.getcwd())
        rel_output_path = os.path.relpath(generated_image_path, start=os.getcwd())
        
        # Create design record in database
        new_design = Design(
            user_id=int(userId),
            promt=textPrompt if mode == "text" else None,
            style=interiorStyle if mode == "quick" else None,
            room_type=roomType if mode == "quick" else None,
            created_at=datetime.utcnow(),
            result_image_path=rel_output_path,
            start_image_path=rel_input_path,
            is_public=is_public
        )
        
        db.add(new_design)
        db.commit()
        db.refresh(new_design)  # Get the ID
        
        # Process furniture recommendations
        image_paths = dict()
        furniture_ids = set()  # To store unique furniture IDs
        for object_key, object_data in search_results.items():
            for item in object_data["similar_items"][:5]:  # Check more items to ensure we have enough unique ones
                if len(furniture_ids) >= 8:  # Maximum of 8 unique furniture items
                    break
                    
                metadata = item["metadata"]
                furniture_id = metadata.get("id_furniture")
                image_path_meta = metadata.get("image")
                image_paths[int(furniture_id)] = image_path_meta
                
                if furniture_id and furniture_id not in furniture_ids:
                    furniture_ids.add(furniture_id)
                    
                    # Create relationship between design and furniture
                    design_furniture = DesignFurniture(
                        design_id=new_design.id, 
                        furniture_id=furniture_id
                    )
                    db.add(design_furniture)
        
        db.commit()
        print(image_paths)
        # Fetch the actual furniture data from the database
        furniture_data = db.query(Furniture).filter(Furniture.id.in_(furniture_ids)).all()
        
        # Prepare furniture recommendations with images
        furniture_recommendations = []
        for furniture in furniture_data:
            # Extract furniture image from the URL or set a placeholder
            image_path = image_paths[furniture.id]
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
                base64_image = base64.b64encode(img_data).decode("utf-8")
                image_paths[furniture.id] = f"data:image/png;base64,{base64_image}"
            
            # Convert price from Decimal to string
            price_str = str(furniture.price)
            
            furniture_item = FurnitureItem(
                id=furniture.id,
                name=furniture.name,
                description=furniture.description,
                price=price_str,
                image=image_paths[furniture.id],  # URL to image on server
                url=furniture.url
            )
            furniture_recommendations.append(furniture_item)
        
        # Convert the generated image to base64 for sending to the frontend
        with open(generated_image_path, "rb") as img_file:
            img_data = img_file.read()
            base64_image = base64.b64encode(img_data).decode("utf-8")
            generated_image_url = f"data:image/png;base64,{base64_image}"
        # Return the response
        return GenerationResponse(
            generatedImage=generated_image_url,
            recommendedFurniture=furniture_recommendations,
            designId=new_design.id
        )
    
    except Exception as e:
        # In case of error, clean up temporary files only - we don't need to delete permanent files
        background_tasks.add_task(cleanup_temp_files, temp_files)
        raise HTTPException(status_code=500, detail=f"Error generating interior design: {str(e)}")


@router.post("/add-furniture")
async def add_furniture(
    photo: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(...),
    type: str = Form(...),
    price: str = Form(...),
    link: str = Form(...),
    partner_id: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        # Просто логируем полученные данные
        print(f"Получен запрос на добавление мебели:")
        print(f"Название: {name}")
        print(f"Описание: {description}")
        print(f"Тип: {type}")
        print(f"Цена: {price}")
        print(f"Ссылка: {link}")
        print(f"ID партнера: {partner_id}")
        print(f"Файл изображения: {photo.filename}")
        
        # Сохраняем загруженное изображение без дальнейшей обработки
        FURNITURE_DIR = Path("./furniture_images")
        FURNITURE_DIR.mkdir(exist_ok=True)
        
        file_name = f"temp_{uuid.uuid4().hex[:8]}_{photo.filename}"
        file_path = FURNITURE_DIR / file_name
        
        # Сохраняем файл только для вида
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)
        
        print(f"Изображение сохранено: {file_path}")
        
        # Без обработки, просто возвращаем успех
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Мебель успешно добавлена (симуляция)",
                "furniture_id": 123,  # Фиктивный ID
                "file_saved": str(file_path)
            }
        )
    
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return JSONResponse(
            status_code=200,  # Все равно возвращаем 200 OK
            content={
                "success": True,
                "message": "Мебель успешно добавлена (симуляция, несмотря на ошибку)",
                "error_details": str(e)
            }
        )

# Health check endpoint
@router.get("/health")
def health_check():
    return {"status": "healthy"}