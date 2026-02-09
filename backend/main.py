from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
# model and shap imports will be added when ml_engine is ready

app = FastAPI(title="SHAP-AID Backend")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    age: float
    bmi: float
    blood_pressure: float
    glucose: float
    insulin: float
    skin_thickness: float
    pregnancies: float
    dpf: float # Diabetes Pedigree Function

@app.get("/")
def read_root():
    return {"message": "SHAP-AID API is running"}

@app.post("/predict")
def predict_risk(data: PatientData):
    try:
        # Convert Pydantic model to dict
        data_dict = data.model_dump()
        
        # Get prediction and explanation
        from ml_engine import engine
        
        risk_score = engine.predict(data_dict)
        explanation = engine.explain(data_dict)
        
        return {
            "risk_score": float(risk_score),
            "risk_class": "High Risk" if risk_score > 0.5 else "Low Risk",
            "shap_values": explanation["breakdown"],
            "base_value": explanation["base_value"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
