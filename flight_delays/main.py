from fastapi import FastAPI
from models import Base
from database import engine, SessionLocal

from starlette.staticfiles import StaticFiles

app = FastAPI()

Base.metadata.create_all(bind=engine)

app.mount('/static', StaticFiles(directory='static'), name='static')


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get('/')
def hello():
    return 'Hi!'