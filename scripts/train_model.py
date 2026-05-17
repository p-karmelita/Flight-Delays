"""Script to train the flight delay prediction model."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_model(data_path: str = "./data/flights.csv", output_path: str = "./models/flight_delay_model.pkl"):
    """
    Train a Linear Regression model for flight delay prediction.
    
    Args:
        data_path: Path to the flights CSV file
        output_path: Path to save the trained model
    """
    print("Loading data...")
    flights = pd.read_csv(data_path, low_memory=False)
    
    print("Preprocessing data...")
    # Select relevant columns
    df = flights[['DEPARTURE_TIME', 'DEPARTURE_DELAY', 'SCHEDULED_TIME', 'ARRIVAL_TIME', 'ARRIVAL_DELAY']].copy()
    
    # Remove missing values
    df.dropna(inplace=True)
    
    print(f"Dataset size after cleaning: {len(df)} rows")
    
    # Prepare features and target
    X = df[['DEPARTURE_TIME', 'DEPARTURE_DELAY', 'SCHEDULED_TIME', 'ARRIVAL_TIME']].values
    y = df['ARRIVAL_DELAY'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    print("Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    print("\n" + "="*50)
    print("Model Performance:")
    print("="*50)
    print(f"Train R² Score: {train_score:.4f}")
    print(f"Test R² Score:  {test_score:.4f}")
    print(f"Train MSE:      {train_mse:.4f}")
    print(f"Test MSE:       {test_mse:.4f}")
    print("="*50)
    
    print(f"\nModel coefficients: {model.coef_}")
    print(f"Model intercept: {model.intercept_}")
    
    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✅ Model saved to: {output_path}")
    
    return model, test_score, test_mse


if __name__ == "__main__":
    import sys
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else "./data/flights.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "./models/flight_delay_model.pkl"
    
    try:
        train_model(data_path, output_path)
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {data_path}")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/gauravmehta13/airline-flight-delays")
        print("\nAnd place it in the data/ directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during training: {e}")
        sys.exit(1)

# Made with Bob
