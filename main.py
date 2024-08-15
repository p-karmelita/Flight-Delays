# from models import Base
# from database import engine
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
import uvicorn


app = FastAPI()

# Base.metadata.create_all(bind=engine)

app.mount('/flight_delays/static', StaticFiles(directory='flight_delays/static'), name='static')
templates = Jinja2Templates(directory='flight_delays/templates')


@app.get('/', response_class=HTMLResponse)
def welcome(request: Request):
    return templates.TemplateResponse('home.html', context={'request': request})


@app.get('/data', response_class=HTMLResponse)
def data(request: Request):
    return templates.TemplateResponse('data.html', context={'request': request})


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
