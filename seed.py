from sqlmodel import SQLModel, Session, select
from database import engine
from models import Tank

SQLModel.metadata.create_all(engine)

tanks = [

    Tank(name="AMX 13", country="France", year=1952,
         description="French light tank with oscillating turret"),

    Tank(name="Fox", country="UK", year=1973,
         description="British reconnaissance vehicle used for scouting missions"),

    Tank(name="Karameh", country="Jordan", year=1968,
         description="Armored vehicle associated with the Battle of Karameh"),

    Tank(name="M42 Duster", country="USA", year=1952,
         description="American self-propelled anti-aircraft gun"),

    Tank(name="M47-Patton", country="USA", year=1951,
         description="American medium tank used during the Cold War"),

    Tank(name="PAK40 Anti-tank gun", country="Germany", year=1942,
         description="German anti-tank gun widely used in WWII"),

    Tank(name="Renault FT-17", country="France", year=1917,
         description="One of the first modern tanks with rotating turret"),

    Tank(name="Rvagnir", country="Sweden", year=1943,
         description="Experimental Swedish armored vehicle"),

    Tank(name="Saladdin", country="UK", year=1958,
         description="British armored car used for reconnaissance"),

    Tank(name="Su", country="USSR", year=1943,
         description="Soviet self-propelled gun used in WWII"),

    Tank(name="The DA Vinci", country="Italy", year=1500,
         description="Leonardo da Vinci conceptual armored vehicle design"),

    Tank(name="Tiger", country="Germany", year=1942,
         description="Heavy German tank used in WWII")

]

with Session(engine) as session:

    for tank in tanks:

        exists = session.exec(
            select(Tank).where(Tank.name == tank.name)
        ).first()

        if not exists:
            session.add(tank)

    session.commit()

print("Database seeded with tanks")