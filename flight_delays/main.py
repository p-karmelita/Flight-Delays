from models import Base
from database import engine
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles


app = FastAPI()

Base.metadata.create_all(bind=engine)

app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')


@app.get('/', response_class=HTMLResponse)
def welcome(request: Request):
    return templates.TemplateResponse('home.html', context={'request': request})


@app.get('/data', response_class=HTMLResponse)
def data(request: Request):
    return templates.TemplateResponse('data.html', context={'request': request})
