import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_diabetes
import pickle
import os

MODEL_PATH = "diabetes_model.pkl"

# Using RandomForest for simplicity and stability with SHAP without xgboost strict dependency
# We can switch to XGBoost if preferred, but RF is great for demo.

class MLEngine:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = [
            "pregnancies", "glucose", "blood_pressure", 
            "skin_thickness", "insulin", "bmi", 
            "dpf", "age"
        ]
        self._load_or_train_model()

    def _load_or_train_model(self):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.explainer = data["explainer"]
            print("Model loaded from disk.")
        else:
            print("Training new model...")
            # Load diabetes dataset - actually let's use a synthetic one or the sklearn one
            # The sklearn load_diabetes is regression. We want classification for risk.
            # I will create a synthetic dataset that mimics the Pima Indians Diabetes structure for the demo.
            
            # Generating synthetic data for training to ensure we have a functional model
            # In a real scenario, we'd load the csv.
            np.random.seed(42)
            n_samples = 1000
            X = pd.DataFrame(np.random.rand(n_samples, 8), columns=self.feature_names)
            # Adjust scales
            X['pregnancies'] = np.random.randint(0, 15, n_samples)
            X['glucose'] = np.random.normal(120, 30, n_samples)
            X['blood_pressure'] = np.random.normal(70, 10, n_samples)
            X['skin_thickness'] = np.random.normal(20, 10, n_samples)
            X['insulin'] = np.random.normal(80, 40, n_samples)
            X['bmi'] = np.random.normal(32, 6, n_samples)
            X['dpf'] = np.random.normal(0.5, 0.3, n_samples)
            X['age'] = np.random.randint(21, 80, n_samples)
            
            # Target: High glucose + high BMI + age -> Risk
            y_prob = (X['glucose'] * 0.4 + X['bmi'] * 0.3 + X['age'] * 0.2) / 200
            y = (y_prob + np.random.normal(0, 0.1, n_samples) > 0.6).astype(int)

            self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            self.model.fit(X, y)

            # Create SHAP explainer
            # KernelExplainer is generic, TreeExplainer is faster for RF
            self.explainer = shap.TreeExplainer(self.model)
            
            with open(MODEL_PATH, "wb") as f:
                pickle.dump({"model": self.model, "explainer": self.explainer}, f)
            print("Model trained and saved.")

    def predict(self, data_dict):
        df = pd.DataFrame([data_dict], columns=self.feature_names)
        prob = self.model.predict_proba(df)[0][1] # Probability of class 1 (High Risk)
        return prob

    def explain(self, data_dict):
        df = pd.DataFrame([data_dict], columns=self.feature_names)
        shap_values = self.explainer.shap_values(df)
        
        # shap_values for binary classification is a list of arrays [class0, class1]
        # or just an array depending on version/model.
        # For RF, it returns list. We want class 1 (Risk).
        if isinstance(shap_values, list):
             sv = shap_values[1][0]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
             # Case where it returns (n_samples, n_features, n_classes) but usually RF is list
             sv = shap_values[0, :, 1]
        else:
             # Regression or binary w/ single output
             if len(shap_values.shape) == 2:
                 sv = shap_values[0] # (n_features,)
             else:
                 sv = shap_values[0]

        # Fix for base_value - robust extraction
        ev = self.explainer.expected_value
        if isinstance(ev, np.ndarray) or isinstance(ev, list):
            if len(np.shape(ev)) > 0 and len(ev) > 1:
                # If we have multiple classes, take the second one (Risk)
                base_value = ev[1]
            else:
                 # Single value (scalar or length-1 array)
                 base_value = ev[0] if len(np.shape(ev)) > 0 else ev
        else:
            base_value = ev

        # Ensure base_value is a simple python float
        if isinstance(base_value, np.ndarray):
            base_value = base_value.item()

        # Construct explanation object
        explanation = []
        for i, feature in enumerate(self.feature_names):
            explanation.append({
                "feature": feature,
                "value": float(df.iloc[0, i]),
                "shap": float(sv[i])
            })
        
        return {
            "base_value": float(base_value),
            "breakdown": explanation
        }

engine = MLEngine()
