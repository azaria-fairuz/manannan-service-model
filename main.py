import os 
import shap
import pickle
import uvicorn
import numpy as np

import onnxruntime as ort

from fastapi import FastAPI
from typing import Optional, Any
from pydantic import BaseModel
from datetime import datetime

from assets import ONNX_BYTES, ARRAY_BYTES, ARRAY_SHAPE, ARRAY_DTYPE, SHAP_BYTES

class SystemResponse(BaseModel):
    status: str
    message: str
    response: Optional[Any] = None

class PredictionData(BaseModel):
    dt: str
    torqa: float = 0.0
    hklda: float = 0.0
    woba: float = 0.0
    rpm: float = 0.0
    mudflowin: float = 0.0
    mudflowoutp: float = 0.0
    stppress: float = 0.0
    t1: int = 0
    t2: int = 0
    t3: int = 0

class PredictionDesc(BaseModel):
    value: float = 0.0
    contrib: float = 0.0

class PredictionResult(BaseModel):
    dt: str
    torqa: PredictionDesc
    hklda: PredictionDesc
    woba: PredictionDesc
    rpm: PredictionDesc
    mudflowin: PredictionDesc
    mudflowoutp: PredictionDesc
    stppress: PredictionDesc
    t1: PredictionDesc
    t2: PredictionDesc
    t3: PredictionDesc

class WellRecords(BaseModel):
    records: list[PredictionData]

train_risk = np.frombuffer(ARRAY_BYTES, dtype=ARRAY_DTYPE).reshape(ARRAY_SHAPE)
ref_sorted = np.sort(train_risk)

def relative_risk(risk):
    return np.searchsorted(ref_sorted, risk, side="righ t") / len(ref_sorted)

def prediction(x):
    x = x.astype(np.float32, copy=False)
    p_risk = session.run([output_name], {input_name: x})[0]
    p_risk = relative_risk(p_risk)
    return p_risk.flatten()

app = FastAPI()

session     = ort.InferenceSession(ONNX_BYTES)
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

shap_backs = pickle.loads(SHAP_BYTES)
explainer  = shap.KernelExplainer(prediction, shap_backs)

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

@app.post("/api/explain", response_model = SystemResponse)
def get_explanation(payload: WellRecords):
    n_feat = np.array([
        [val for val in row.model_dump().values() if isinstance(val, (int, float))]
        for row in payload.records
    ], dtype = np.float32)
    p_risk = prediction(n_feat)
    p_vals = explainer.shap_values(n_feat)
    p_risk = p_risk.tolist()
    p_data = payload.records

    results = []
    for i in range(len(p_data)):
        row_dict = {"dt": p_data[i].dt}
        full_dump = p_data[i].model_dump()
        numeric_keys = [k for k, v in full_dump.items() if isinstance(v, (int, float))]
        
        for idx, key in enumerate(numeric_keys):
            row_dict[key] = PredictionDesc(
                value=float(full_dump[key]),
                contrib=round(p_vals[i][idx], 3)
            )
            
        result = PredictionResult(**row_dict)
        results.append(result)

    return {
        "status" : "success",
        "message" : "stuck pipe risk has been successfully predicted",
        "response" : results
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5106))
    uvicorn.run(app, host="0.0.0.0", port=port)