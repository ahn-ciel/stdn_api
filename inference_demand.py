import torch
import numpy as np
import json
from types import SimpleNamespace
from utils import infer_load_data
from models import STDN
from trainer_v2 import Trainer
import datetime
import logging

def load_args_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    value_name = data.get("VALUE", "")
    # "VALUE" 문자열을 모두 value_name으로 치환
    def replace_value(obj):
        if isinstance(obj, str):
            return obj.replace("VALUE", value_name)
        elif isinstance(obj, dict):
            return {k: replace_value(v) for k, v in obj.items() if k != "VALUE"}
        elif isinstance(obj, list):
            return [replace_value(i) for i in obj]
        else:
            return obj

    replaced_data = replace_value(data)
    return SimpleNamespace(**replaced_data)

def setup_logger(log_file):
    # Setup logger configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s]: %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Log to file
            logging.StreamHandler()         # Log to console
        ]
    )
    
# 0. 엔진 준비
def initialize_engine(args):
    # 디바이스 설정
    multi_gpu = False
    if args.gpu is not None:
        if isinstance(args.gpu, list) and len(args.gpu) > 1:
            args.device = torch.device(f'cuda:{args.gpu[0]}')
            torch.cuda.set_device(args.gpu[0])
            multi_gpu = True
        else:
            args.device = torch.device(f'cuda:{args.gpu[0]}' if isinstance(args.gpu, list) else f'cuda:{args.gpu}')
            torch.cuda.set_device(args.gpu[0] if isinstance(args.gpu, list) else args.gpu)
    else:
        args.device = torch.device("cpu")

    model = STDN(args)
    logging.info(f"Using device: {args.device}")

    # 다중 GPU 처리
    if multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)

    state_dict = torch.load(args.model_path, map_location="cpu")
    if any(k.startswith("module.") for k in state_dict.keys()):
        logging.info("Detected DataParallel model. Removing 'module.' prefix...")
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    else:
        logging.info("Detected Single-GPU model.")
        model.load_state_dict(state_dict)

    return model

# 1. 모델만 로드
def setup_model(args):
    # 로그 설정
    log_file = f"/STDN/logs/infer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # log_file = args.save + f'_infer_{today}_{args.model_path.split("/")[-1].split(".")[0]}.log'
    setup_logger(log_file)

    logging.info(args)

    model = initialize_engine(args)
    return model

# 2. 데이터만 로드하는 함수
def load_data(args):
    data = infer_load_data(args)

    infer_loader = torch.utils.data.DataLoader(
        data['infer_dataset'],
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.gpu is not None,
        num_workers=4
    )
    # [!주의!] data['infer_dataset'].start_ind +1: 383+1개 미만이면 예측 불가
    print("-----len(data):",len(data))
    if len(data['infer_dataset'].volume) < data['infer_dataset'].start_ind + 1:
        # raise ValueError(
        #     f"Not enough data for inference. Required: {data['infer_dataset'].start_ind + 1}, "
        #     f"but got {len(data['infer_dataset'].volume)}")
        logging.warning(f"Not enough data for inference. Required: {data['infer_dataset'].start_ind + 1}, "
            f"but got {len(data['infer_dataset'].volume)}")    

    logging.info(f"Total samples: {len(data['infer_dataset'].volume)}, Start index: {data['infer_dataset'].start_ind}")
    return data, infer_loader

def predict(args, model, data, infer_loader):
    
    args.threshold = float(args.threshold_ / args.volume_train_max)
    trainer = Trainer(model, args, test_loader=infer_loader)
    
    # 추론 및 결과 저장
    time_info = data['infer_dataset'].get_data_time()
    output = trainer.test()
    output = output * args.volume_train_max
    print("result.shape:",output.shape)
    y = data['infer_dataset'].volume[data['infer_dataset'].start_ind:] * args.volume_train_max

    # post-process
    output = np.clip(output, 0, None) # 음수 값을 0    
    output = np.rint(output).astype(int) # 가까운 정수
    y = np.rint(y).astype(int)
    # np.savez_compressed(f"{args.save_dir}/results_{config['name']}_{time_info[0]}.npz",
    #         prediction = output,
    #         ground_truth = y,
    #         time=time_info)
    
    time_expanded =time_info.reshape(-1, 1, 1, 1)
    time_broadcasted = np.tile(time_expanded, (1, output.shape[1], output.shape[2], 1))  # (7, 10, 20, 1)
    print(output.shape, y.shape, time_broadcasted.shape)
    combind = np.concatenate([output, y, time_broadcasted], axis=-1)
    print(combind.shape)

    date_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    api_response = {
        "run_time": date_now,
        "timestamps": time_info,
        "prediction": output,
        "ground_truth": y
    }

    return api_response

if __name__ == "__main__":
    
    # craate logfile
    # log_file = f"/STDN/logs/inferDemand_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # setup_logger(log_file)
    
    config_path = '/STDN/data_dj_infer.json' 
    args = load_args_from_json(config_path)

    model = setup_model(args)
    data, infer_loader = load_data(args)
    result = predict(args, model, data, infer_loader)
    
    print(result["prediction"])
    print(result["timestamps"])
    print(result["prediction"].shape, result["ground_truth"].shape)
    