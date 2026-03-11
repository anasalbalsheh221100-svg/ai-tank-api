from sqlmodel import SQLModel, Session
from database import engine
from models import Tank

# إنشاء الجداول أولاً
SQLModel.metadata.create_all(engine)

tanks = [
    Tank(name="Tiger", country="Germany", year=1942,
         description="Heavy German tank used in WWII"),

    Tank(name="AMX 13", country="France", year=1952,
         description="French light tank with oscillating turret"),

    Tank(name="M47-Patton", country="USA", year=1951,
         description="American medium tank used in the Cold War")
]

with Session(engine) as session:
    for tank in tanks:
        session.add(tank)

    session.commit()