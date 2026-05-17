"""Main FastAPI application."""

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import __version__
from app.config import get_settings
from app.database import Base, engine
from app.ml_model import ml_model
from app.schemas import FlightInput, HealthResponse, PredictionResponse

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    Base.metadata.create_all(bind=engine)
    ml_model.load_model()
    yield
    # Shutdown
    pass


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API for predicting flight arrival delays using machine learning",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="flight_delays/static"), name="static")
templates = Jinja2Templates(directory="flight_delays/templates")


@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def home(request: Request):
    """Render home page."""
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "version": __version__,
        },
    )


@app.get("/data", response_class=HTMLResponse, tags=["Frontend"])
async def data_page(request: Request):
    """Render data visualization page."""
    return templates.TemplateResponse("data.html", {"request": request})


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        model_loaded=ml_model.is_loaded,
    )


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_delay(flight_input: FlightInput):
    """
    Predict flight arrival delay.
    
    This endpoint uses a trained Linear Regression model to predict
    the arrival delay based on departure time, departure delay,
    scheduled time, and arrival time.
    """
    if not ml_model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs.",
        )

    try:
        # Prepare features for prediction
        features = np.array(
            [
                flight_input.departure_time,
                flight_input.departure_delay,
                flight_input.scheduled_time,
                flight_input.arrival_time,
            ]
        )

        # Make prediction
        predicted_delay = ml_model.predict(features)

        return PredictionResponse(
            predicted_arrival_delay=round(predicted_delay, 2),
            model_score=0.89,  # This should be loaded from model metadata
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}",
        )


@app.get("/api/model-info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    if not ml_model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    return ml_model.get_model_info()

# Made with Bob
