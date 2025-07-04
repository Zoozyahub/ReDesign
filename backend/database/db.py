from sqlalchemy import (
    Column, String, Integer, BigInteger, Text, Boolean, ForeignKey,
    DateTime, Numeric
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


Base = declarative_base()

class User(Base):
    __tablename__ = "user"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    last_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    phone = Column(String(255), unique=True, nullable=False)
    password = Column(Text, nullable=False)
    role = Column(Integer, nullable=False)  # Лучше сделать enum в будущем
    created_at = Column(DateTime, default=datetime.utcnow)

    designs = relationship("Design", back_populates="user")

class Design(Base):
    __tablename__ = "design"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("user.id"), nullable=False)
    promt = Column(Text)
    style = Column(String(255))
    room_type = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    result_image_path = Column(String(255), nullable=False)
    start_image_path = Column(String(255), nullable=False)
    is_public = Column(Boolean, default=False, nullable=False)
    feedback = Column(Text, nullable=True)

    user = relationship("User", back_populates="designs")
    furniture = relationship("DesignFurniture", back_populates="design")

class Furniture(Base):
    __tablename__ = "furniture"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    type = Column(String(255), nullable=False)
    price = Column(Numeric(10, 2), nullable=False)
    url = Column(Text, nullable=False)
    partner_id = Column(BigInteger, ForeignKey("partner.id"), nullable=False)

    partner = relationship("Partner", back_populates="furnitures")
    designs = relationship("DesignFurniture", back_populates="furniture")

class DesignFurniture(Base):
    __tablename__ = "design_furniture"

    design_id = Column(BigInteger, ForeignKey("design.id"), primary_key=True)
    furniture_id = Column(BigInteger, ForeignKey("furniture.id"), primary_key=True)

    design = relationship("Design", back_populates="furniture")
    furniture = relationship("Furniture", back_populates="designs")

class Partner(Base):
    __tablename__ = "partner"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    legal_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    url = Column(String(255), nullable=False)
    logo_url = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    phone = Column(String(255), nullable=False)

    furnitures = relationship("Furniture", back_populates="partner")

def get_db():
    DATABASE_URL = "postgresql://postgres:1234@localhost:5432/interior_db"

    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Создание таблиц
def create_db():
    DATABASE_URL = "postgresql://postgres:1234@localhost:5432/interior_db"
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    print("Успешно")  
    
    
# Функция для вставки данных в таблицу furniture
def insert_furniture_data(db: Session):
    furniture_data = [
        {"id": 1, "name": "Серое кресло", "description": "без описания", "type": "armchair", "price": 1200, "url": "/home", "partner_id": 1},
        {"id": 2, "name": "Кожанное кресло", "description": "без описания", "type": "armchair", "price": 1500, "url": "/home", "partner_id": 1},
        {"id": 3, "name": "Цветное кресло", "description": "без описания", "type": "armchair", "price": 2100, "url": "/home", "partner_id": 1},
        {"id": 4, "name": "Жёлтое кресло", "description": "без описания", "type": "armchair", "price": 3000, "url": "/home", "partner_id": 1},
        {"id": 5, "name": "Чёрный небольшой диван", "description": "без описания", "type": "couch", "price": 12000, "url": "/home", "partner_id": 1},
        {"id": 6, "name": "Белый небольшой диван", "description": "без описания", "type": "couch", "price": 12000, "url": "/home", "partner_id": 1},
        {"id": 7, "name": "Маленький деревянный столик", "description": "без описания", "type": "table", "price": 3500, "url": "/home", "partner_id": 1},
        {"id": 8, "name": "Кровать с подъёмным механизмом Victori цвет тёмно-серый 160х200 см",
         "description": "Victori — кровать удобна и элегантна. Ее каркас выполнен из прочной и долговечной ДСП, обивка — из мягкого бархатистого велюра. Привлекает внимание высокая мягкая спинка в изголовье: прострочка в виде диагональных линий, образующих в центре буквы V, выглядит оригинально и эффектно.",
         "type": "bed", "price": 23999,
         "url": "https://hoff.ru/catalog/spalnya/krovati/krovati_s_podemnym_mehanizmom/krovat_s_podyemnym_mekhanizmom_victori_id10408649/?articul=80642640 ", "partner_id": 2},
        {"id": 9, "name": "Каркас кровати Глазго цвет таксония, металл бруклин",
         "description": "Двуспальная кровать Глазго отлично подойдет для спальни, выполненной в современном стиле. Модель отличается высокой комфортностью и оригинальным исполнением. В ее дизайне сочетаются различные цвета и фактуры. Мягкое стеганое изголовье обито серой искусственной кожей, приятной на ощупь и устойчивой к истиранию.",
         "type": "bed", "price": 10999,
         "url": "https://hoff.ru/catalog/spalnya/krovati/krovati_bez_podemnogo_mehanizma/karkas_krovati_glazgo_id8282207/?articul=80500498 ", "partner_id": 2},
        {"id": 10, "name": "Кровать Хелен цвет дуб крафт золотой, серый графит 160х200 см", "description": "без описания", "type": "bed", "price": 11999,
         "url": "https://hoff.ru/catalog/spalnya/krovati/krovati_bez_podemnogo_mehanizma/krovat_khelen_id10063049/?articul=80621558 ", "partner_id": 2},
        {"id": 11, "name": "Кресло SCANDICA Скотт", "description": "без описания", "type": "armchair", "price": 15999,
         "url": "https://hoff.ru/catalog/gostinaya/kresla/kreslo_scandica_skott_id8851657/?articul=80538589 ", "partner_id": 2},
        {"id": 12, "name": "Диван Хангель", "description": "без описания", "type": "couch", "price": 39999,
         "url": "https://hoff.ru/catalog/gostinaya/divany/pryamye/divan_khangel_id8951451/?articul=80543041 ", "partner_id": 2},
        {"id": 13, "name": "Диван Астер-М", "description": "без описания", "type": "couch", "price": 29999,
         "url": "https://hoff.ru/catalog/gostinaya/divany/pryamye/divan_aster_m_id7870882/?articul=80423900 ", "partner_id": 2},
        {"id": 14, "name": "Диван-кровать Хэппи", "description": "без описания", "type": "couch", "price": 59999,
         "url": "https://hoff.ru/catalog/gostinaya/divany/pryamye/divan_krovat_kheppi_id10109025/?articul=80622885 ", "partner_id": 2},
    ]

    for data in furniture_data:
        furniture = Furniture(**data)
        db.add(furniture)

    db.commit()
    print("Данные успешно добавлены в таблицу furniture.")

def insert_partner_data(db: Session):
    furniture_data = [
        {"id": 1, "name": "Partner", "legal_name": 'Partner', "description": "без описания", "url": "/home", "logo_url": '/home', 'email': 'part@mail.ru', 'phone': '88005553535'},
        {"id": 2, "name": "Hoff", 'legal_name': 'Hoff', "description": "главный эксперт в мебели и товарах для дома", "url": "https://hoff.ru/", "logo_url": 'https://p0.zoon.ru/0/9/5d1dd069cb9c6a5560386302_5d1dd0fc52930.jpg', 'email': 'hoff@mail.ru', 'phone': '89093406666'},
    ]

    for data in furniture_data:
        furniture = Partner(**data)
        db.add(furniture)

    db.commit()
    print("Данные успешно добавлены в таблицу partner.")

# Вызов функции вставки данных
if __name__ == "__main__":
    # # Подключение к базе данных и вставка данных
    # DATABASE_URL = "postgresql://postgres:1234@localhost:5432/interior_db"
    # engine = create_engine(DATABASE_URL)
    # SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    # db = SessionLocal()

    # # Вставка данных
    # insert_partner_data(db)
    # insert_furniture_data(db)
    pass