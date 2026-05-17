"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, Field


class FlightInput(BaseModel):
    """Schema for flight prediction input."""

    departure_time: float = Field(
        ...,
        description="Departure time in minutes from midnight (0-1439)",
        ge=0,
        le=1439,
    )
    departure_delay: float = Field(
        ...,
        description="Departure delay in minutes",
        ge=-100,
        le=1000,
    )
    scheduled_time: float = Field(
        ...,
        description="Scheduled flight time in minutes",
        ge=0,
        le=1000,
    )
    arrival_time: float = Field(
        ...,
        description="Arrival time in minutes from midnight (0-1439)",
        ge=0,
        le=1439,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "departure_time": 900.0,
                    "departure_delay": 15.0,
                    "scheduled_time": 120.0,
                    "arrival_time": 1020.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""

    predicted_arrival_delay: float = Field(
        ...,
        description="Predicted arrival delay in minutes",
    )
    model_score: float = Field(
        ...,
        description="Model R² score on test data",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_arrival_delay": 18.5,
                    "model_score": 0.89,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str
    version: str
    model_loaded: bool

# Made with Bob
