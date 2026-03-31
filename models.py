from sqlmodel import SQLModel, Field

class Tank(SQLModel, table=True):

    id: int | None = Field(default=None, primary_key=True)

    name: str = Field(index=True)
    country: str
    year: int
    description: str