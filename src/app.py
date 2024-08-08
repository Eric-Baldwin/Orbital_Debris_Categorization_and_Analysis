from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load the model
with open('best_logreg_model.pkl', 'rb') as f:
    model = pickle.load(f)

class PredictionRequest(BaseModel):
    input: list

class PredictionResponse(BaseModel):
    prediction: list

@app.get("/")
def read_root():
    return {"message": "Welcome to the Space Debris Analysis API"}

@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        data = np.array(request.input).reshape(1, -1)
        prediction = model.predict(data)
        return PredictionResponse(prediction=prediction.tolist())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
