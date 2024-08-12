from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load the model, polynomial features transformer, and scaler
with open('best_logreg_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('poly_transformer.pkl', 'rb') as f:
    poly_transformer = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

class PredictionRequest(BaseModel):
    inclination: float
    period_hours: float
    altitude_mi: float
    eccentricity: float
    ra_of_asc_node: float
    arg_of_pericenter: float
    mean_anomaly: float
    semimajor_axis: float
    apoapsis: float
    periapsis: float
    rcs_size: int

class PredictionResponse(BaseModel):
    object_type: str
    size_classification: str

@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Extract input features as an array
        input_data = np.array([[
            request.inclination, request.period_hours, request.altitude_mi,
            request.eccentricity, request.ra_of_asc_node, request.arg_of_pericenter,
            request.mean_anomaly, request.semimajor_axis, request.apoapsis, request.periapsis
        ]])

        # Apply polynomial features transformation
        input_poly = poly_transformer.transform(input_data)

        # Standardize the features
        input_standardized = scaler.transform(input_poly)

        # Make prediction for object type
        object_type_prediction = model.predict(input_standardized)

        # Determine object type
        object_type = "Payload" if object_type_prediction[0] == 1 else "Non-Payload"

        # Determine size classification based on RCS_SIZE
        size_classification_map = {0: "Small", 1: "Medium", 2: "Large"}
        size_classification = size_classification_map.get(request.rcs_size, "Unknown")

        return PredictionResponse(
            object_type=object_type,
            size_classification=size_classification
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
