from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from train_demand  import main, create_save_dir, load_args_from_json, setup_logging
from typing import List
from fastapi import Query
from fastapi.encoders import jsonable_encoder
import numpy as np
import copy

# app = FastAPI()
app = FastAPI(default_response_class=ORJSONResponse)

@app.on_event("startup")
def startup_event():
    # 설정 파일 경로
    config_path = "/STDN/data_dj_train.json"
    args = load_args_from_json(config_path)

    # 로그 디렉토리 및 로그 파일 설정
    create_save_dir(args)

    app.state.args = args
    # app.state.log_file = log_file
    
@app.get("/train-demand", response_class=ORJSONResponse)
def train_with_config():
    args_base = app.state.args
    args = copy.deepcopy(args_base)
    # args = load_args_from_json("/STDN/data_dj_train.json")
    # create_save_dir(args)
    result = main(args)

    # NumPy → JSON serializable
    return jsonable_encoder(result, custom_encoder={np.ndarray: lambda x: x.tolist()})