from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from inference_demand  import predict, load_args_from_json, setup_model, load_data, setup_logger
from typing import List
from fastapi.encoders import jsonable_encoder
import numpy as np
import datetime
import os
import copy

# app = FastAPI()
app = FastAPI(default_response_class=ORJSONResponse)

@app.on_event("startup")
def startup_event():
    # 설정 파일 경로
    config_path = "/STDN/data_dj_infer.json"
    args = load_args_from_json(config_path)

    # # 로그 디렉토리 및 로그 파일 설정
    # log_dir = "/STDN/logs"
    # os.makedirs(log_dir, exist_ok=True)
    # log_file = os.path.join(log_dir, f"infer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    # setup_logger(log_file)

    # 모델 로딩 및 log 저장 
    model = setup_model(args)

    app.state.args = args
    app.state.model = model

    
@app.get("/predict-demand", response_class=ORJSONResponse)
def predict_demand():
    args = copy.deepcopy(app.state.args)

    # 데이터 및 로더 로딩
    data, infer_loader = load_data(args)

    # 추론 실행
    result = predict(args, app.state.model, data, infer_loader)

    # NumPy → JSON serializable
    return jsonable_encoder(result, custom_encoder={np.ndarray: lambda x: x.tolist()})