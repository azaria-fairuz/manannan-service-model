import os 
import uvicorn
import numpy as np

import onnxruntime as ort

from fastapi import FastAPI
from typing import Optional, Any
from pydantic import BaseModel
from datetime import datetime

from assets import ONNX_BYTES, ARRAY_BYTES, ARRAY_SHAPE, ARRAY_DTYPE

class SystemResponse(BaseModel):
    status: str
    message: str
    response: Optional[Any] = None

class PredictionData(BaseModel):
    dt: datetime
    torqa: float
    hklda: float
    woba: float
    rpm: float
    mudflowin: float
    mudflowoutp: float
    stppress: float
    t1: float
    t2: float
    t3: float

class WellRecords(BaseModel):
    records: list[PredictionData]

app = FastAPI()

session     = ort.InferenceSession(ONNX_BYTES)
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

train_risk = np.frombuffer(ARRAY_BYTES, dtype=ARRAY_DTYPE).reshape(ARRAY_SHAPE)
ref_sorted = np.sort(train_risk)

@app.get("/api", response_model = SystemResponse)
def index():
    return {
        "status" : "success",
        "message" : "model api is alive and well",
        "response" : []
    }

@app.post("/api/prediction", response_model = SystemResponse)
def get_prediction(payload: WellRecords):
    n_feat = np.array([
        [val for val in row.model_dump().values() if isinstance(val, (int, float))]
        for row in payload.records
    ], dtype = np.float32)
    p_risk = session.run([output_name], {input_name: n_feat})[0]
    p_risk = relative_risk(p_risk)
    p_risk = p_risk.flatten().tolist()

    p_data = payload.records
    for i in range(len(p_data)):
        p_data[i] = dict(p_data[i])
        p_data[i]['risk'] = round(p_risk[i], 3)
        p_data[i]['alert'] = int(p_risk[i] > 0.97)

    return {
        "status" : "success",
        "message" : "stuck pipe risk has been successfully predicted",
        "response" : p_data
    }

def relative_risk(risk):
    return np.searchsorted(ref_sorted, risk, side="righ t") / len(ref_sorted)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5106))
    uvicorn.run(app, host="0.0.0.0", port=port)