import sys
import json
import numpy as np
import joblib
import logging
import os
from pathlib import Path
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_model_files(model_path, scaler_path):
    """Validate that model files exist and are not corrupted."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    
    try:
        # Try to load the files to check if they're not corrupted
        joblib.load(model_path)
        joblib.load(scaler_path)
    except Exception as e:
        raise ValueError(f"Corrupted model or scaler file: {str(e)}")

def load_model_and_scaler():
    try:
        # Get the absolute path to the model and scaler files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'knn_model.sav')
        scaler_path = os.path.join(current_dir, 'scaler.sav')
        
        # Validate files before loading
        validate_model_files(model_path, scaler_path)
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Loading scaler from: {scaler_path}")
        
        # Load the model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model or scaler: {str(e)}")
        raise

def validate_input_features(features):
    """Validate input features against expected ranges."""
    expected_ranges = {
        'N': (0, 140),
        'P': (0, 140),
        'K': (0, 140),
        'temperature': (0, 50),
        'humidity': (0, 100),
        'ph': (0, 14),
        'rainfall': (0, 300)
    }
    
    for feature, value in features.items():
        if feature not in expected_ranges:
            raise ValueError(f"Unexpected feature: {feature}")
        
        min_val, max_val = expected_ranges[feature]
        if not (min_val <= value <= max_val):
            raise ValueError(f"{feature} value {value} is outside valid range [{min_val}, {max_val}]")

def preprocess_input(features, scaler):
    try:
        # Convert features to numpy array and reshape for prediction
        # Create a DataFrame to maintain feature names
        features_df = pd.DataFrame([features])
        
        # Scale the features
        features_scaled = scaler.transform(features_df)
        return features_scaled
    except Exception as e:
        logger.error(f"Error preprocessing input: {str(e)}")
        raise

def predict_crop(features):
    try:
        # Validate input features
        validate_input_features(features)
        
        # Load model and scaler
        model, scaler = load_model_and_scaler()
        
        # Preprocess input
        features_scaled = preprocess_input(features, scaler)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        # Get prediction probability
        probabilities = model.predict_proba(features_scaled)
        
        # Get top 3 predictions with their probabilities
        top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
        top_3_crops = model.classes_[top_3_indices]
        top_3_probs = probabilities[0][top_3_indices]
        
        # Create result dictionary
        result = {
            "predicted_crop": prediction[0],
            "confidence": float(probabilities[0][np.argmax(probabilities[0])]),
            "top_3_recommendations": [
                {"crop": crop, "probability": float(prob)}
                for crop, prob in zip(top_3_crops, top_3_probs)
            ]
        }
        
        logger.info(f"Prediction successful: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def main():
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())
        
        # Validate required features
        required_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        if not all(feature in input_data for feature in required_features):
            missing = [f for f in required_features if f not in input_data]
            raise ValueError(f"Missing required features: {missing}")
        
        # Convert features to float
        features = {}
        for key, value in input_data.items():
            try:
                features[key] = float(value)
            except ValueError:
                raise ValueError(f"Invalid value for {key}: {value}")
        
        # Make prediction
        result = predict_crop(features)
        
        # Output result as JSON
        print(json.dumps(result))
        
    except json.JSONDecodeError:
        logger.error("Invalid JSON input")
        print(json.dumps({"error": "Invalid JSON input"}))
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Input validation error: {str(e)}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(json.dumps({"error": "An unexpected error occurred"}))
        sys.exit(1)

if __name__ == "__main__":
    main() 