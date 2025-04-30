import os, json, time
from types import SimpleNamespace
from models import STDN
from trainer_v2 import Trainer
from utils import *
import torch
import numpy as np
import logging
import datetime

def create_save_dir(args):
    if os.path.exists(args.save_dir):
        reply = str(input(f'{args.save_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit()
    else:
        os.makedirs(args.save_dir)
        
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
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    
    # save_path : 동적으로 추가
    if "save_dir" in replaced_data and today not in replaced_data['save_dir']:
        replaced_data['save_dir'] = os.path.join(replaced_data['save_dir'], today)
    
    if "save_dir" in replaced_data and "name" in replaced_data:
        replaced_data["save_path"] = os.path.join(replaced_data["save_dir"], f"STDN_{replaced_data['name']}.pth")
            
    return SimpleNamespace(**replaced_data)

def setup_logging(log_file):
    # Setup logger configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s]: %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Log to file
            logging.StreamHandler()         # Log to console
        ]
    )

def main(args):
    # config = json.load(open(args.config, 'r'))
    # Setup logger to log into a file and the console
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    log_file = f"{args.save_dir}/train_{args.name}_{today}_log.txt"
    setup_logging(log_file)
    
    if args.seed is not None:
        logging.info(f"Start Deterministic Training with seed {args.seed}")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    multi_gpu = False
    if args.gpu is not None:
        if isinstance(args.gpu, list) and len(args.gpu) > 1:
            args.device = torch.device(f'cuda:{args.gpu[0]}')  # 첫 번째 GPU를 메인으로 설정
            torch.cuda.set_device(args.gpu[0])  # 메인 GPU 설정
            multi_gpu = True
        else:
            args.device = torch.device(f'cuda:{args.gpu[0]}' if isinstance(args.gpu, list) else f'cuda:{args.gpu}')
            torch.cuda.set_device(args.gpu[0] if isinstance(args.gpu, list) else args.gpu)
    else:
        args.device = 'cpu'
    
    logging.info(args)
    logging.info(f"Using device: {args.device}")
    
    # load data 
    data = train_load_data(args)
    
    train_time = data['train_dataset'].get_data_time()
    data_size = len(data['train_dataset'])
    indices = list(range(data_size))
    split = int(data_size * 0.2) # 0.2 == validation_split
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_timeinfo = [train_time[i] for i in train_indices]
    val_timeinfo = [train_time[i] for i in val_indices]
    print("train_indices, val_indices:", len(train_indices), len(val_indices))
    # DataLoader 최적화: 멀티 GPU 사용 시 `num_workers` 증가
    num_workers = len(args.gpu) * 2 if multi_gpu else 4
    
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    print(len(train_sampler), len(val_sampler))
    train_loader = torch.utils.data.DataLoader(data['train_dataset'], batch_size = args.batch_size, shuffle = False, pin_memory = args.gpu is not None, sampler = train_sampler,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(data['train_dataset'], batch_size = args.batch_size, shuffle = False, pin_memory = args.gpu is not None, sampler = val_sampler,
                                             num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(data['test_dataset'], batch_size = args.batch_size, shuffle = False, pin_memory = args.gpu is not None,
                                              num_workers=num_workers)
    
    logging.info(f"Total train/ val/ test samples: {sum(1 for _ in train_loader.dataset)}, {sum(1 for _ in val_loader.dataset)}, {sum(1 for _ in test_loader.dataset)}")    
    
    # load model
    model = STDN(args)

    if multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
    model.to(args.device)
    
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    args.threshold = float(args.threshold_ / args.volume_train_max)
    trainer = Trainer(model, args, train_loader, val_loader, test_loader)
        
    if args.evaluate:
        mse, mape, rmse = trainer.eval(val = False)
    else:
        save_model_path =trainer.train()        
        trainer.model.load_state_dict(torch.load(save_model_path))
        mse, mape, rmse = trainer.eval(val = False)
    
    logging.info(f"After training, Test MSE: {args.volume_train_max * mse:.4f}, MAPE: {mape:.3f}, RMSE: {args.volume_train_max * rmse:.3f}")
    
    # ahn: 시간 정보 가져오기
    time_info = data['test_dataset'].get_data_time()
    results = trainer.test()
    results = results * args.volume_train_max
    y = data['test_dataset'].volume[data['test_dataset'].start_ind:] * args.volume_train_max
    
    save_eval_path = f"{args.save_dir}/STDN_results_{args.name}_{results.shape[0]:02d}.npz"
    np.savez_compressed(save_eval_path, prediction = results, ground_truth = y, time=time_info)
    api_response={
        "run_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message1": f"Trained model is stored : {save_model_path}",
        "message2": f"Results of given test data have been saved in .npz format. File path: {save_eval_path}"
    }    
    return api_response


if __name__ == "__main__":
    t1 = time.time()
    config_path = '/STDN/data_dj_train.json' 
    args = load_args_from_json(config_path)
    
    create_save_dir(args)
    
    api_response=main(args)    
    print(api_response)
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', default = 8, type = int, help = "Training batch size")
    # parser.add_argument('--epochs', default = 1000, type = int, help = "Maximum epochs to train model")
    # # parser.add_argument('--learning_rate', default = 1e-3, type = float, help = "Learning rate")
    # # parser.add_argument('--config', default = 'data_dj.json', type = str, help = "Configuration file path")
    # # parser.add_argument('--vol_in', default = 2, type = int, help = "input volume dimension, default: 2 (volume_in, volume_out)")
    # # parser.add_argument('--flow_in', default = 4, type = int, help = "input flow dimension, default: 4")
    # # parser.add_argument('--cnn_channels', default = 64, type = int, help = "Hidden dimension for the CNNs")
    # # parser.add_argument('--flat_size', default = 128, type = int, help = "Hidden dimension for the flat layer after CNNs")
    # # parser.add_argument('--kernel_size', default = 3, type = int, help = "Kernel size for the CNNs")
    # # parser.add_argument('--num_layer', default = 3, type = int, help = "Number of Local CNN layers")
    # # parser.add_argument('--nbhd_size', default = 7, type = int, help = "Patch size for the LocalCNN, default: 7 (CNN on 7x7 grids)")
    # # parser.add_argument('--att_lstm_num', default = 3, type = int, help = "Number of days for the attention lstm default: 3 (eprevious 3 days)")
    # # parser.add_argument('--lstm_out_size', default = 128, type = int, help = "Hidden size for the LSTM")
    # # parser.add_argument('--long_term_lstm_seq_len', default = 3, type = int, help = "LSTM sequence length for the previous days")
    # # parser.add_argument('--short_term_lstm_seq_len', default = 7, type = int, help = "LSTM sequence length for the recent data (e.g., 3.5 hours)")
    # # parser.add_argument('--hist_feature_daynum', default = 7, type = int, help = "historical daynum for the external data (e.g., 7 days)")
    # # parser.add_argument('--last_feature_num', default = 48, type = int, help = "last feature number for the external data (48 * 30 minutes interval == 24 hours)")
    # parser.add_argument('--seed', default = 99, type = int, help = "For debugging. If provided, model will be trained with given random seed")
    # parser.add_argument('--save_dir', default = f"checkpoints/{today}", type = str, help = "Save path for the model")
    # parser.add_argument('--load_path', default = None, type = str, help = "Loading path for the pre-trained model")
    # parser.add_argument('--evaluate', action = 'store_true', help = "Flag for the evaluation. If flag == True, it will evaluate given model without training")
    # parser.add_argument('--print_every', default = 10, type = int, help = "Printing intervals")
    # parser.add_argument('--gpu', default = None, type = int, nargs='+', help = "GPU ID for the training and testing. -1 means cpu")
    # args = parser.parse_args()
    # # args.ext_size = args.vol_in * (args.last_feature_num + args.hist_feature_daynum)
    # main(args)
    """
    •	att_lstm_num: attention LSTM 개수 (기본값: 3)
	•	long_term_lstm_seq_len: 장기 LSTM 시퀀스 길이 (기본값: 3)
	•	short_term_lstm_seq_len: 단기 LSTM 시퀀스 길이 (기본값: 7)
	•	hist_feature_daynum: 과거 며칠간의 데이터를 사용할지 결정 (기본값: 7일)
	•	last_feature_num: 과거 몇 개의 타임슬롯을 사용할지 결정 (기본값: 48개, 즉 하루)
	•	nbhd_size: LSTM에서 사용하는 주변 지역 크기 (기본값: 1)
	•	cnn_nbhd_size: CNN에서 사용하는 주변 지역 크기 (기본값: 3)
    """