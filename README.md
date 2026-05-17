![FlightDelays](https://assets.planespotters.net/files/user/profile/78/c1/78c1c06b-331e-4add-a276-1e7b1ed6166f_256.png)

# ✈️ Flight Delays Predictor

A modern machine learning web application for predicting flight arrival delays using Linear Regression. Built with FastAPI, scikit-learn, and Docker.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.5-009688.svg)](https://fastapi.tiangolo.com)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Project Overview

This data science project uses a Linear Regression model to predict arrival delays from a dataset of 5,000,000+ commercial airline flights from 2015[^1]. The model achieves an R² score of 0.89 on test data.

### Features

- 🚀 **RESTful API** - FastAPI-based prediction endpoint with automatic documentation
- 🎨 **Interactive Web UI** - User-friendly interface for making predictions
- 🐳 **Docker Support** - Containerized deployment with Docker Compose
- 🧪 **Comprehensive Tests** - Unit and integration tests with pytest
- 📈 **Data Visualization** - Built-in data analysis and visualization pages
- 🔒 **Type Safety** - Full Pydantic validation for API requests/responses
- ⚙️ **Configuration Management** - Environment-based configuration with .env support

## 🏗️ Architecture

```
Flight-Delays/
├── app/                    # Main application package
│   ├── __init__.py
│   ├── main.py            # FastAPI application
│   ├── config.py          # Configuration management
│   ├── database.py        # Database setup
│   ├── models.py          # SQLAlchemy models
│   ├── schemas.py         # Pydantic schemas
│   └── ml_model.py        # ML model wrapper
├── flight_delays/         # Legacy code (templates & static files)
│   ├── static/           # CSS, JS, images
│   └── templates/        # HTML templates
├── tests/                # Test suite
│   ├── test_api.py
│   └── test_models.py
├── models/               # Trained ML models
├── data/                 # Dataset files
├── .github/workflows/    # CI/CD pipelines
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/p-karmelita/Flight-Delays.git
   cd Flight-Delays
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

6. **Access the application**
   - Web UI: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker compose up --build
   ```

2. **Access the application**
   - Application: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

> **Note for Windows users:** If you encounter connection issues, try accessing `http://127.0.0.1:8000` instead of `http://0.0.0.0:8000`

## 📡 API Usage

### Predict Flight Delay

**Endpoint:** `POST /api/predict`

**Request Body:**
```json
{
  "departure_time": 900.0,
  "departure_delay": 15.0,
  "scheduled_time": 120.0,
  "arrival_time": 1020.0
}
```

**Response:**
```json
{
  "predicted_arrival_delay": 18.5,
  "model_score": 0.89
}
```

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "model_loaded": true
}
```

### Model Information

**Endpoint:** `GET /api/model-info`

**Response:**
```json
{
  "loaded": true,
  "coefficients": [0.123, 0.456, 0.789, 0.012],
  "intercept": 1.234
}
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## 🛠️ Development

### Code Quality

The project uses several tools to maintain code quality:

```bash
# Format code with Black
black app tests

# Sort imports with isort
isort app tests

# Lint with Flake8
flake8 app tests --max-line-length=100

# Type checking with mypy
mypy app --ignore-missing-imports
```

### Pre-commit Setup

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## 📦 Technologies

- **Backend Framework:** FastAPI 0.115.5
- **ML Library:** scikit-learn 1.5.2
- **Data Processing:** pandas 2.2.3, numpy 2.0.2
- **Database:** SQLAlchemy 2.0.36
- **Visualization:** matplotlib 3.9.2, seaborn 0.13.2
- **Testing:** pytest 8.3.4
- **Code Quality:** black, flake8, mypy, isort
- **Containerization:** Docker, Docker Compose
- **CI/CD:** GitHub Actions

## 📈 Model Performance

- **Algorithm:** Linear Regression
- **R² Score:** 0.89
- **MSE:** 161.16
- **Training Data:** 5M+ flight records from 2015
- **Features:** Departure time, departure delay, scheduled time, arrival time

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Piotr Karmelita**

- GitHub: [@p-karmelita](https://github.com/p-karmelita)

## 🙏 Acknowledgments

- Dataset source: [Kaggle - Airline Flight Delays](https://www.kaggle.com/datasets/gauravmehta13/airline-flight-delays)[^1]
- Built with [FastAPI](https://fastapi.tiangolo.com/)
- ML powered by [scikit-learn](https://scikit-learn.org/)

---

[^1]: [✈️ Dataset: Airline Flight Delays 2015](https://www.kaggle.com/datasets/gauravmehta13/airline-flight-delays)
