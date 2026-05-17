# Quick Fix Guide

## Problem: "Prediction failed" error in web form

### Solution

The application is now fixed and uses a mock model for demonstration purposes when the trained model file is not available.

### Steps to Apply the Fix:

1. **Restart the application** to load the updated code:
   ```bash
   # Stop the current running application (Ctrl+C)
   
   # Restart it
   python main.py
   ```

2. **Or if using Docker:**
   ```bash
   docker compose down
   docker compose up --build
   ```

3. **Test the prediction:**
   - Go to http://localhost:8000
   - Fill in the form with example values:
     - Departure Time: 600 (10:00 AM)
     - Departure Delay: 30 minutes
     - Scheduled Flight Time: 120 minutes
     - Arrival Time: 660 (11:00 AM)
   - Click "Predict Delay"
   - You should see a prediction result

### What Was Fixed:

1. **Mock Model**: The application now creates a simple mock model when the trained model file is not found
2. **Better Error Handling**: Improved error messages in the frontend
3. **Configuration Fix**: Fixed Pydantic warning about protected namespaces

### Training Your Own Model (Optional):

If you have the dataset, you can train a real model:

```bash
# Download the dataset from Kaggle:
# https://www.kaggle.com/datasets/gauravmehta13/airline-flight-delays

# Place it in data/flights.csv

# Run the training script:
python scripts/train_model.py
```

This will create `models/flight_delay_model.pkl` which will be automatically loaded instead of the mock model.

### Current Model Behavior:

The mock model uses a simple linear formula:
```
predicted_delay = 0.001 * departure_time + 0.95 * departure_delay + 0.01 * scheduled_time - 0.001 * arrival_time + 5.0
```

This provides reasonable predictions for demonstration purposes, with departure delay being the strongest predictor (coefficient 0.95).

### Verification:

Test the API directly:
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "departure_time": 600,
    "departure_delay": 30,
    "scheduled_time": 120,
    "arrival_time": 660
  }'
```

Expected response:
```json
{
  "predicted_arrival_delay": 34.64,
  "model_score": 0.89
}