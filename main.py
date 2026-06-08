import os 
import uvicorn
import numpy as np

import onnxruntime as ort

from fastapi import FastAPI
from typing import Optional, Any
from pydantic import BaseModel

from assets import ONNX_BYTES, ARRAY_BYTES, ARRAY_SHAPE, ARRAY_DTYPE

class SystemResponse(BaseModel):
    status: str
    message: str
    response: Optional[Any] = None

class PredictionData(BaseModel):
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
def get_prediction(records: PredictionData):
    n_feat = np.array(list(records.model_dump().values()))
    n_feat = np.expand_dims(n_feat.astype(np.float32), axis = 0)
    p_risk = session.run(
        [output_name], 
        {input_name: n_feat}
    )[0]
    p_risk = relative_risk(p_risk)[0][0]

    return {
        "status" : "success",
        "message" : "stuck pipe risk has been successfully predicted",
        "response" : {
            "risk"  : p_risk,
            "alert" : int(p_risk > 0.97),
            "data"  : records
        }
    }

def relative_risk(risk):
    return np.searchsorted(ref_sorted, risk, side="right") / len(ref_sorted)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5106))
    uvicorn.run(app, host="0.0.0.0", port=port)