from pathlib import Path
from sqlmodel import create_engine, Session

BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = f"sqlite:///{BASE_DIR}/tank.db"

engine = create_engine(DATABASE_URL)

def get_session():
    with Session(engine) as session:
        yield session