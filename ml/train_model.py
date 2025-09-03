import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_and_save_model():
    try:
        # Load the dataset
        logger.info("Loading dataset...")
        dataset_path = r"C:\Users\Welcome\OneDrive\Desktop\Chandru\LIL\Mark5\ml\Crop_Recommendation.csv"    
        logger.info(f"Dataset path: {dataset_path}")
        
        # Try to read the CSV file
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Successfully loaded dataset with columns: {df.columns.tolist()}")
            logger.info(f"Dataset shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            raise
        
        # Prepare features and target
        try:
            # Map the column names to our expected feature names
            column_mapping = {
                'Nitrogen': 'N',
                'Phosphorus': 'P',
                'Potassium': 'K',
                'Temperature': 'temperature',
                'Humidity': 'humidity',
                'pH_Value': 'ph',
                'Rainfall': 'rainfall'
            }
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
            y = df['Crop']  # Target column is 'Crop'
            logger.info(f"Features shape: {X.shape}")
            logger.info(f"Unique labels: {np.unique(y)}")
        except Exception as e:
            logger.error(f"Error preparing features and target: {str(e)}")
            raise
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        logger.info("Training KNN model...")
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)
        
        # Save the model and scaler
        logger.info("Saving model and scaler...")
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knn_model.sav')
        scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaler.sav')
        logger.info(f"Saving model to: {model_path}")
        logger.info(f"Saving scaler to: {scaler_path}")
        
        # Save model and scaler with proper error handling
        try:
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            logger.info("Successfully saved model and scaler")
        except Exception as e:
            logger.error(f"Error saving model or scaler: {str(e)}")
            raise
        
        # Evaluate the model
        accuracy = model.score(X_test_scaled, y_test)
        logger.info(f"Model accuracy: {accuracy:.2f}")
        
        return model, scaler
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model() 