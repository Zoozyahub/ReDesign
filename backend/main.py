from fastapi import FastAPI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from fastapi.middleware.cors import CORSMiddleware

from api.auth import router as auth_router
from api.generate import router as generate_router
from api.gallery import router as gallery_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить запросы с любых адресов
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы
    allow_headers=["*"],  # Разрешить все заголовки
)

app.include_router(auth_router, prefix="/api")
app.include_router(generate_router, prefix="/api")
app.include_router(gallery_router, prefix='/api')
    
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    