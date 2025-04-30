import numpy as np
import os, json
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    """
    Input
     - volume --> shape (B, H, W, args.vol_in)
     - flow --> shape (B, H, W, S, S, args.flow_in)
     - config --> see data.json
     - args --> see main.py
    We will get input as below:  
     - att_cnn_x 9, (B, H, W, 2)                          
     - att_flow 9, (B, H, W, S, S, 4)                     
     - att_x 3, (B, H, W, 3, 160) or 3, (B*H*W, 3, 160)   
     - cnn_x 7, (B, H, W, 2)                              
     - flow 7, (B, H, W, S, S, 4)                         
     - x (B, H, W, 7, 160) or (B*H*W, 7, 160)             
     - y (B, H, W, 2)                                     
    """
    def __init__(self, config, volume, flow, dataTime, args, \
        hist_feature_daynum=7, att_lstm_num=3, long_term_lstm_seq_len=3, short_term_lstm_seq_len=7, last_feature_num=48):
        args.att_lstm_num = att_lstm_num
        args.long_term_lstm_seq_len = long_term_lstm_seq_len
        args.short_term_lstm_seq_len = short_term_lstm_seq_len
        args.hist_feature_daynum = hist_feature_daynum
        args.last_feature_num= last_feature_num
        self.config = config
        self.timeslot_daynum = int(86400 / config["timeslot_sec"])
        self.start_ind = (hist_feature_daynum + att_lstm_num) * self.timeslot_daynum + long_term_lstm_seq_len
        self.volume = volume.astype(np.float32)
        self.flow = flow.astype(np.float32)
        self.H = self.volume.shape[1]
        self.W = self.volume.shape[2]
        self.args = args
        
        self.dataTime = np.array(dataTime)  # time 저장 (하지만 학습에는 사용하지 않음)

    def __getitem__(self, index):
        """
        vol: B, H, W, 2, flow B, H, W, S, S, 4
        
        Attention 데이터 - 과거 장기 데이터
        att_cnn_x	(B, H, W, 2) : 과거 CNN 입력 데이터
        att_flow	(B, H, W, S, S, 4) : 과거 Flow 입력 데이터
        att_x	(B, H, W, 3, 160) or (B*H*W, 3, 160) : 과거 LSTM Attention 입력 데이터
        
        Short-term 및 Long-term Feature 생성 : 과거 데이터를 활용하여 생성
        hist	(B, H, W, 7, 160) : 과거 7일(hist_feature_daynum) 데이터를 기반
        last	(B, H, W, 7, 160) : 최근 last_feature_num개의 데이터를 기반
        x	(B, H, W, 7, 160) : LSTM 모델 입력 데이터
        
        CNN 및 Flow 입력 데이터 - 단기
        cnn_x	(B, H, W, 2) : CNN 모델 입력 데이터 (과거 short-term 데이터)
        flow	(B, H, W, S, S, 4) : 과거 short-term의 Flow 입력 데이터
        y	(B, H, W, 2) : 현재 시간의 volume 데이터를 예측 대상(Y 값)으로 설정
        
        cur_index보다 작은 모든 데이터가 과거 데이터로 활용
        """
        cur_index = index + self.start_ind 
        # print("self.start_ind,index,cur_index:",self.start_ind,index,cur_index)
        att_start = int(cur_index - (self.args.att_lstm_num) * self.timeslot_daynum - self.args.long_term_lstm_seq_len // 2)
        att_range = range(att_start, cur_index - self.timeslot_daynum, self.timeslot_daynum)
        att_cnn_x = []
        att_flow = []
        att_x = []
        for att_t in att_range:
            att_cnn_x.append(self.volume[att_t:att_t+self.args.long_term_lstm_seq_len])
            att_flow.append(self.flow[att_t:att_t+self.args.long_term_lstm_seq_len])
            att_hist = np.stack([self.volume[t-self.args.hist_feature_daynum*self.timeslot_daynum:t:self.timeslot_daynum].transpose((1,2,0,3)).reshape(self.H, self.W, -1)\
                                 for t in range(att_t, att_t+self.args.long_term_lstm_seq_len)], axis = -2)
            att_last = np.stack([self.volume[t-self.args.last_feature_num:t].transpose((1,2,0,3)).reshape(self.H, self.W, -1)\
                                 for t in range(att_t, att_t+self.args.long_term_lstm_seq_len)], axis = -2)
            att_x.append(np.concatenate([att_hist, att_last], axis = -1))

        att_cnn_x = np.concatenate(att_cnn_x, axis = 0)
        att_flow = np.concatenate(att_flow, axis = 0)
        att_x = np.stack(att_x, axis = 0)
        cnn_x = self.volume[cur_index - self.args.short_term_lstm_seq_len:cur_index]
        flow = self.flow[cur_index - self.args.short_term_lstm_seq_len:cur_index]
        hist = np.stack([self.volume[t-self.args.hist_feature_daynum*self.timeslot_daynum:t:self.timeslot_daynum].transpose((1,2,0,3)).reshape(self.H, self.W, -1)\
                         for t in range(cur_index - self.args.short_term_lstm_seq_len, cur_index)], axis = -2)
        last = np.stack([self.volume[t-self.args.last_feature_num:t].transpose((1,2,0,3)).reshape(self.H, self.W, -1)\
                         for t in range(cur_index - self.args.short_term_lstm_seq_len, cur_index)], axis = -2)

        x = np.concatenate([hist, last], axis = -1)
        y = self.volume[cur_index]
        
        return att_cnn_x, att_flow, att_x, cnn_x, flow, x, y
    
    def get_data_time(self):
        """ 시간 정보를 반환하는 함수 (예측 결과 비교 시 사용) """
        return self.dataTime[self.start_ind:]  # start_ind 이후의 time 정보만 반환

    def __len__(self):
        return self.volume.shape[0] - self.start_ind

class CustomDataset_v2(Dataset):
    """
    Input
     - volume --> shape (B, H, W, args.vol_in)
     - flow --> shape (B, H, W, S, S, args.flow_in)
     - config --> see data.json
     - args --> see main.py
    We will get input as below:  
     - att_cnn_x 9, (B, H, W, 2)                          
     - att_flow 9, (B, H, W, S, S, 4)                     
     - att_x 3, (B, H, W, 3, 160) or 3, (B*H*W, 3, 160)   
     - cnn_x 7, (B, H, W, 2)                              
     - flow 7, (B, H, W, S, S, 4)                         
     - x (B, H, W, 7, 160) or (B*H*W, 7, 160)             
     - y (B, H, W, 2)                                     
    """
    def __init__(self, volume, flow, dataTime, args, \
        hist_feature_daynum=7, att_lstm_num=3, long_term_lstm_seq_len=3, short_term_lstm_seq_len=7, last_feature_num=48):
        args.att_lstm_num = att_lstm_num
        args.long_term_lstm_seq_len = long_term_lstm_seq_len
        args.short_term_lstm_seq_len = short_term_lstm_seq_len
        args.hist_feature_daynum = hist_feature_daynum
        args.last_feature_num= last_feature_num
        self.timeslot_daynum = int(86400 / args.timeslot_sec)
        self.start_ind = (hist_feature_daynum + att_lstm_num) * self.timeslot_daynum + long_term_lstm_seq_len
        self.volume = volume.astype(np.float32)
        self.flow = flow.astype(np.float32)
        self.H = self.volume.shape[1]
        self.W = self.volume.shape[2]
        self.args = args
        
        self.dataTime = np.array(dataTime)  # time 저장 (하지만 학습에는 사용하지 않음)

    def __getitem__(self, index):
        """
        vol: B, H, W, 2, flow B, H, W, S, S, 4
        
        Attention 데이터 - 과거 장기 데이터
        att_cnn_x	(B, H, W, 2) : 과거 CNN 입력 데이터
        att_flow	(B, H, W, S, S, 4) : 과거 Flow 입력 데이터
        att_x	(B, H, W, 3, 160) or (B*H*W, 3, 160) : 과거 LSTM Attention 입력 데이터
        
        Short-term 및 Long-term Feature 생성 : 과거 데이터를 활용하여 생성
        hist	(B, H, W, 7, 160) : 과거 7일(hist_feature_daynum) 데이터를 기반
        last	(B, H, W, 7, 160) : 최근 last_feature_num개의 데이터를 기반
        x	(B, H, W, 7, 160) : LSTM 모델 입력 데이터
        
        CNN 및 Flow 입력 데이터 - 단기
        cnn_x	(B, H, W, 2) : CNN 모델 입력 데이터 (과거 short-term 데이터)
        flow	(B, H, W, S, S, 4) : 과거 short-term의 Flow 입력 데이터
        y	(B, H, W, 2) : 현재 시간의 volume 데이터를 예측 대상(Y 값)으로 설정
        
        cur_index보다 작은 모든 데이터가 과거 데이터로 활용
        """
        cur_index = index + self.start_ind 
        # print("self.start_ind,index,cur_index:",self.start_ind,index,cur_index)
        att_start = int(cur_index - (self.args.att_lstm_num) * self.timeslot_daynum - self.args.long_term_lstm_seq_len // 2)
        att_range = range(att_start, cur_index - self.timeslot_daynum, self.timeslot_daynum)
        att_cnn_x = []
        att_flow = []
        att_x = []
        for att_t in att_range:
            att_cnn_x.append(self.volume[att_t:att_t+self.args.long_term_lstm_seq_len])
            att_flow.append(self.flow[att_t:att_t+self.args.long_term_lstm_seq_len])
            att_hist = np.stack([self.volume[t-self.args.hist_feature_daynum*self.timeslot_daynum:t:self.timeslot_daynum].transpose((1,2,0,3)).reshape(self.H, self.W, -1)\
                                 for t in range(att_t, att_t+self.args.long_term_lstm_seq_len)], axis = -2)
            att_last = np.stack([self.volume[t-self.args.last_feature_num:t].transpose((1,2,0,3)).reshape(self.H, self.W, -1)\
                                 for t in range(att_t, att_t+self.args.long_term_lstm_seq_len)], axis = -2)
            att_x.append(np.concatenate([att_hist, att_last], axis = -1))

        att_cnn_x = np.concatenate(att_cnn_x, axis = 0)
        att_flow = np.concatenate(att_flow, axis = 0)
        att_x = np.stack(att_x, axis = 0)
        cnn_x = self.volume[cur_index - self.args.short_term_lstm_seq_len:cur_index]
        flow = self.flow[cur_index - self.args.short_term_lstm_seq_len:cur_index]
        hist = np.stack([self.volume[t-self.args.hist_feature_daynum*self.timeslot_daynum:t:self.timeslot_daynum].transpose((1,2,0,3)).reshape(self.H, self.W, -1)\
                         for t in range(cur_index - self.args.short_term_lstm_seq_len, cur_index)], axis = -2)
        last = np.stack([self.volume[t-self.args.last_feature_num:t].transpose((1,2,0,3)).reshape(self.H, self.W, -1)\
                         for t in range(cur_index - self.args.short_term_lstm_seq_len, cur_index)], axis = -2)

        x = np.concatenate([hist, last], axis = -1)
        y = self.volume[cur_index]
        
        return att_cnn_x, att_flow, att_x, cnn_x, flow, x, y
    
    def get_data_time(self):
        """ 시간 정보를 반환하는 함수 (예측 결과 비교 시 사용) """
        return self.dataTime[self.start_ind:]  # start_ind 이후의 time 정보만 반환

    def __len__(self):
        return self.volume.shape[0] - self.start_ind

def get_local_data(local_data, x, y, nbhd_size):
    """
    B, H, W, 4
    """
    offsets = nbhd_size // 2
    data = np.pad(local_data,
            pad_width = ((0,0), (offsets, offsets), (offsets, offsets), (0,0)),
            mode = 'constant')
    nbhd_data = data[:,x:x+2*offsets+1, y:y+2*offsets+1]
    return nbhd_data

def process_flow(flow_data, nbhd_size=7, flow_input=4):
    """
    flow shape (2, B, 10, 20, 10, 20)
    change them into: (B, 10, 20, 4), where:
     - 2, B, x, y, 10, 20 and
     - 2, B, 10, 20, x, y
    """
    H, W = flow_data.shape[-2:]
    processed_data = []
    for x in range(H):
        for y in range(W):
            flow_out = flow_data[:,:,x,y] # 2, B, H, W
            flow_in = flow_data[:,:,:,:, x, y]
            local_data = np.stack([flow_out[0], flow_in[0], flow_out[1], flow_in[1]], axis = -1) # B, H, W, 4
            local_data = get_local_data(local_data, x, y, nbhd_size) # B, S, S, 4
            processed_data.append(local_data)
    processed_data = np.stack(processed_data, axis = 1)

    processed_data = processed_data.reshape(-1, H, W, nbhd_size, nbhd_size, flow_input)
    return processed_data


def load_data(config, args):
    """
    volume shape (B, H, W, 2)
    하루(86400초)를 timeslot_sec (30분 = 1800초)로 나눈 개수 -> 하루에 48개 타임스텝이 생성
    """
    timeslot_daynum = int(86400 / config["timeslot_sec"])
    threshold = int(config["threshold"])
    datas = {}
    for cat in 'volume_train volume_test flow_train flow_test'.split():
        _type = cat.split('_')[0]
        # np.load("/STDN/data/ciel/djOD_volume_train.npz")['volume'] / config[f"volume_train_max"]
        data = np.load(config[cat])[_type] / config[f"{_type}_train_max"]
        # ahn : add timesptemp
        dataTime = np.load(config[cat])['time']
        if 'flow' == _type:
            data = process_flow(data)
        datas[cat] = data
        # ahn
        datas[cat + '_time'] = dataTime  # time 저장 (학습에는 사용 안 함)

    for cat in 'train test'.split():
        # datas[cat + '_dataset'] = CustomDataset(config, datas['volume_'+cat], datas['flow_'+cat], args)
        datas[cat + '_dataset'] = CustomDataset(config, 
                                                datas['volume_'+cat], 
                                                datas['flow_'+cat], 
                                                datas['volume_'+cat+'_time'],  # time 전달
                                                args)

    return datas

def train_load_data(args):
    """
    volume shape (B, H, W, 2)
    하루(86400초)를 timeslot_sec (30분 = 1800초)로 나눈 개수 -> 하루에 48개 타임스텝이 생성
    """
    timeslot_daynum = int(86400 / args.timeslot_sec)
    threshold = int(args.threshold_)
    datas = {}
    for cat in 'volume_train volume_test flow_train flow_test'.split():
        _type = cat.split('_')[0]
        # np.load("/STDN/data/ciel/djOD_volume_train.npz")['volume'] / config[f"volume_train_max"]
        testData = getattr(args, f"{cat}")
        max_value = getattr(args, f"{_type}_train_max")
        data = np.load(testData)[_type] / max_value
        # ahn : add timesptemp
        dataTime = np.load(testData)['time']
        if 'flow' == _type:
            data = process_flow(data)
        datas[cat] = data
        # ahn
        datas[cat + '_time'] = dataTime  # time 저장 (학습에는 사용 안 함)

    for cat in 'train test'.split():
        datas[cat + '_dataset'] = CustomDataset_v2(datas['volume_'+cat], 
                                                datas['flow_'+cat], 
                                                datas['volume_'+cat+'_time'],  # time 전달
                                                args)

    return datas

def infer_load_data(args):
    timeslot_daynum = int(86400 / args.timeslot_sec)
    threshold = int(args.threshold_)
    datas = {}
    for cat in 'volume_test flow_test'.split():
        _type = cat.split('_')[0]
        testData = getattr(args, f"{cat}")
        max_value = getattr(args, f"{_type}_train_max")
        data = np.load(testData)[_type] / max_value
        # ahn : add timesptemp
        dataTime = np.load(testData)['time']
        if 'flow' == _type:
            data = process_flow(data)
        datas[cat] = data
        # ahn
        datas[cat + '_time'] = dataTime  # time 저장 (학습에는 사용 안 함)    
    datas['infer_dataset'] = CustomDataset_v2(datas['volume_test'], 
                                            datas['flow_test'], 
                                            datas['volume_test_time'],  # time 전달
                                            args)           
    return datas


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vol_in', default = 2)
    parser.add_argument('--flow_in', default = 4)
    parser.add_argument('--cnn_channels', default = 64)
    parser.add_argument('--flat_size', default = 128)
    parser.add_argument('--kernel_size', default = 3)
    parser.add_argument('--num_layer', default = 3)
    parser.add_argument('--nbhd_size', default = 7)
    parser.add_argument('--att_lstm_num', default = 3)
    parser.add_argument('--lstm_out_size', default = 128)
    parser.add_argument('--long_term_lstm_seq_len', default = 3)
    parser.add_argument('--short_term_lstm_seq_len', default = 3)
    parser.add_argument('--hist_feature_daynum', default = 7)
    parser.add_argument('--last_feature_num', default = 48)
    args = parser.parse_args()
    args.ext_size = args.vol_in * (args.last_feature_num + args.hist_feature_daynum)
    import json
    config = json.load(open('data.json', 'r'))
    data = load_data(config, args)
    att_cnn_x, att_flow, att_x, cnn_x, flow, x, y = data['test_dataset'][0]
    print(att_cnn_x.shape)
    print(att_flow.shape)
    print(att_x.shape)
    print(cnn_x.shape)
    print(flow.shape)
    print(x.shape)
    print(y.shape)
