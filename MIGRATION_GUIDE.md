# Migration Guide - Flight Delays v1 → v2

This guide helps you migrate from the old project structure to the new modernized version.

## 🔄 Major Changes

### 1. Project Structure

**Old Structure:**
```
Flight-Delays/
├── main.py
├── flight_delays/
│   ├── main.py (duplicate)
│   ├── models.py
│   ├── database.py
│   └── flight_delays.py (messy analysis code)
```

**New Structure:**
```
Flight-Delays/
├── main.py (entry point only)
├── app/ (main application)
│   ├── main.py (FastAPI app)
│   ├── config.py (configuration)
│   ├── models.py (DB models)
│   ├── schemas.py (Pydantic schemas)
│   ├── database.py (DB setup)
│   └── ml_model.py (ML wrapper)
├── tests/ (test suite)
└── flight_delays/ (legacy - templates & static only)
```

### 2. Configuration Management

**Old:** Hardcoded values in code
```python
SQLALCHEMY_DATABASE_URL = 'sqlite:///./flights.db'
```

**New:** Environment-based configuration
```python
# .env file
DATABASE_URL=sqlite:///./flights.db
DEBUG=False
HOST=0.0.0.0
PORT=8000
```

### 3. API Endpoints

**New Endpoints Added:**
- `POST /api/predict` - Make predictions via API
- `GET /health` - Health check endpoint
- `GET /api/model-info` - Get model information
- `GET /docs` - Auto-generated API documentation

### 4. Dependencies

**Updated to latest versions:**
- FastAPI: 0.112.0 → 0.115.5
- Pydantic: 2.8.2 → 2.10.3
- SQLAlchemy: 2.0.31 → 2.0.36
- pandas: 2.2.2 → 2.2.3
- scikit-learn: 1.4.1 → 1.5.2

**Removed unnecessary packages:**
- Flask (not used)
- APScheduler (not used)
- nltk (not used)
- Many other unused dependencies

### 5. Docker Configuration

**Old Dockerfile:**
- Single-stage build
- No health checks
- Basic setup

**New Dockerfile:**
- Multi-stage build (smaller image)
- Health checks included
- Optimized for production

## 📋 Migration Steps

### Step 1: Backup Your Data

```bash
# Backup your database
cp flights.db flights.db.backup

# Backup your model if you have one
cp model.pkl models/flight_delay_model.pkl
```

### Step 2: Update Dependencies

```bash
# Remove old virtual environment
rm -rf .venv

# Create new virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install new dependencies
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
nano .env  # or use your preferred editor
```

### Step 4: Update Import Statements

If you have custom code, update imports:

**Old:**
```python
from models import Base
from database import engine
```

**New:**
```python
from app.models import Flight
from app.database import engine, get_db
from app.schemas import FlightInput, PredictionResponse
```

### Step 5: Run Tests

```bash
# Run the test suite to ensure everything works
pytest tests/ -v
```

### Step 6: Start the Application

```bash
# Local development
python main.py

# Or with Docker
docker compose up --build
```

## 🔧 Breaking Changes

### 1. Model File Location

**Old:** `model.pkl` in root directory
**New:** `models/flight_delay_model.pkl`

**Action:** Move your model file:
```bash
mkdir -p models
mv model.pkl models/flight_delay_model.pkl
```

### 2. Database Model Changes

The `Flights` model now has an `id` field as primary key:

```python
# Old
class Flights(Base):
    arrival_delay = Column(Float, primary_key=True)  # Bad practice

# New
class Flight(Base):
    id = Column(Integer, primary_key=True, autoincrement=True)
    arrival_delay = Column(Float, nullable=False)
```

### 3. API Response Format

Predictions now return structured JSON:

```json
{
  "predicted_arrival_delay": 18.5,
  "model_score": 0.89
}
```

### 4. Static Files Path

**Old:** `/static/...`
**New:** `/static/...` (unchanged, but now properly configured)

## 🆕 New Features

### 1. Interactive Prediction Form

The home page now includes a form to make predictions directly from the browser.

### 2. API Documentation

Access auto-generated API docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. Health Checks

Monitor application health:
```bash
curl http://localhost:8000/health
```

### 4. Type Safety

All API endpoints now have Pydantic validation:
```python
class FlightInput(BaseModel):
    departure_time: float = Field(..., ge=0, le=1439)
    departure_delay: float = Field(..., ge=-100, le=1000)
    # ...
```

### 5. Testing Infrastructure

Comprehensive test suite with pytest:
```bash
pytest --cov=app --cov-report=html
```

### 6. CI/CD Pipeline

GitHub Actions workflow for:
- Code linting (black, flake8, isort, mypy)
- Running tests
- Building Docker images

## 🐛 Common Issues

### Issue 1: Module Not Found

**Error:** `ModuleNotFoundError: No module named 'app'`

**Solution:** Make sure you're running from the project root:
```bash
cd /path/to/Flight-Delays
python main.py
```

### Issue 2: Model Not Loading

**Error:** `Model file not found`

**Solution:** Ensure model is in correct location:
```bash
ls models/flight_delay_model.pkl
```

### Issue 3: Database Connection Error

**Error:** `Could not connect to database`

**Solution:** Check your `.env` file:
```bash
DATABASE_URL=sqlite:///./flights.db
```

### Issue 4: Port Already in Use

**Error:** `Address already in use`

**Solution:** Change port in `.env`:
```bash
PORT=8001
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Docker Documentation](https://docs.docker.com/)
- [pytest Documentation](https://docs.pytest.org/)

## 🤝 Need Help?

If you encounter issues during migration:
1. Check the logs for error messages
2. Review the updated README.md
3. Open an issue on GitHub
4. Contact the maintainer

---

**Last Updated:** 2026-05-17