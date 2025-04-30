import torch
import torch.nn as nn
import torch.nn.functional as F

"""
vol_in: input volume dimension, default= 2 (volume_in, volume_out)
flow_in: input flow dimension, default= 4
cnn_channels: Hidden dimension for the CNNs, default= 64
flat_size: Hidden dimension for the flat layer after CNNs, default= 128
kernel_size: Kernel size for the CNNs, default= 3
num_layer: Number of Local CNN layers, default= 3
nbhd_size: Patch size for the LocalCNN, default= 7 (CNN on 7x7 grids)
att_lstm_num: Number of days for the attention lstm, default= 3 (eprevious 3 days)
lstm_out_size: Hidden size for the LSTM, default= 128
long_term_lstm_seq_len: LSTM sequence length for the previous days, default= 3
short_term_lstm_seq_len: LSTM sequence length for the recent data, default= 7 (e.g., 3.5 hours)
hist_feature_daynum: historical daynum for the external data, default= 7 (e.g., 7 days) 
last_feature_num: last feature number for the external data, default= 48 (48 * 30 minutes interval == 24 hours)

•	att_lstm_num: attention LSTM 개수 (기본값: 3), 주의 메커니즘의 반복 횟수 (논문의 Q 값)
•	long_term_lstm_seq_len: 장기 LSTM 시퀀스 길이 (기본값: 3), (=att_lstm_seq_len)주의 메커니즘에서 고려하는 날짜 수 (논문의 P 값)
•	short_term_lstm_seq_len: 단기 LSTM 시퀀스 길이 (기본값: 7), LSTM 시퀀스 길이
•	hist_feature_daynum: 외부 요인(날씨, 이벤트 등) 고려를 위한 과거 며칠간의 데이터를 사용할지 결정 (기본값: 7일)
•	last_feature_num: 과거 몇 개의 타임슬롯을 사용할지 결정 (기본값: 48개, 즉 하루)
•	nbhd_size: LSTM에서 사용하는 주변 지역 크기 (기본값: 7)
•	cnn_nbhd_size: CNN에서 사용하는 주변 지역 크기 (기본값: 3)
"""
class STDN(nn.Module):
    # def __init__(self, args):
    def __init__(self, args, vol_in=2, flow_in=4, cnn_channels=64, flat_size=128, kernel_size=3, \
        num_layer=3, nbhd_size=7, att_lstm_num=3, lstm_out_size=128, long_term_lstm_seq_len=7, \
        hist_feature_daynum=7, last_feature_num=48, learning_rate= 1e-3):        
        super(STDN, self).__init__()
        args.flat_size = flat_size
        args.att_lstm_num = att_lstm_num
        args.vol_in = vol_in
        self.args = args
        self.ext_size = vol_in * (last_feature_num + hist_feature_daynum)
        self.local_cnn = StackedLocalConv()
        self.cur_lstm = ext_lstm(self.ext_size, is_cur = True)
        self.attn_lstms = nn.ModuleList([ext_lstm(self.ext_size, is_cur = False) for _ in range(att_lstm_num)])
        self.psam = PSAM(lstm_out_size=128, bias = False)
        self.fc = nn.Linear(lstm_out_size * 2, vol_in)

    def forward(self, att_cnn_x, att_flow, att_x, cnn_x, flow, x):
        """
        Original input/output
        att_cnn_x 9 (95400, 7, 7, 2) --> 9: att_lstm_num * long_term_lstm_seq_len
        att_flow 9 (95400, 7, 7, 4) --> 9: att_lstm_num * long_term_lstm_seq_len
        att_x 3 (95400, 3, 160) --> front 3: att_lstm_num, back 3: long_term_lstm_seq_len
        cnn_x 7 (95400, 7, 7, 2) --> 7: short_term_lstm_seq_len
        flow 7 (95400, 7, 7, 4) --> 7: short_term_lstm_seq_len
        x (95400, 7, 160) --> 7: short_term_lstm_seq_len
        y (95400, 2)

        Modified input/output
        We will get input as below:
         - att_cnn_x 9, (B, H, W, 2)
         - att_flow 9, (B, H, W, S, S, 4)
         - att_x 3, (B, H, W, 3, 160) or 3, (B*H*W, 3, 160)
         - cnn_x 7, (B, H, W, 2)
         - flow 7, (B, H, W, S, S, 4)
         - x (B, H, W, 7, 160) or (B*H*W, 7, 160)
         - y (B, H, W, 2)
         Notes
          - 95400 --> B * H * W
          - volume, flow --> make_local_patches in StackedLocalConv
          - att_x, x --> change shapes only (e.g., B, H, W, 2 --> B*H*W, 2, if we needs)
          - y --> Will remain same

        lstm_out --> B, lstm_out_size
        """   
        B, _, H, W, _ = cnn_x.size()
        cnn_x = cnn_x.transpose(0,1).flatten(0,1)
        flow = flow.transpose(0,1).flatten(0,1)
        x = x.flatten(0,2)
        att_cnn_x = att_cnn_x.transpose(0,1).flatten(0,1)
        att_flow = att_flow.transpose(0,1).flatten(0,1)
        att_x = att_x.transpose(0,1).flatten(1,3)


        cnn_out = self.local_cnn(cnn_x, flow)
        lstm_out = self.cur_lstm(cnn_out.reshape(-1, B*H*W, self.args.flat_size).permute(1,0,2), x)

        att_cnn_out = self.local_cnn(att_cnn_x, att_flow).reshape(-1, B * H * W, self.args.flat_size).permute(1, 0, 2)
        att_cnn_outs = att_cnn_out.chunk(self.args.att_lstm_num, dim = 1)
        att_lstm_outs = []

        for i, lstm in enumerate(self.attn_lstms):
            att_lstm_outs.append(lstm(att_cnn_outs[i], att_x[i]))

        attn_out = self.psam(lstm_out, att_lstm_outs)

        output = torch.tanh(self.fc(torch.cat([attn_out, lstm_out], dim = -1)))
        output = output.reshape(B, H, W, self.args.vol_in)
        return output

class StackedLocalConv(nn.Module):
    """
    Local Convolustion w/ flow and volume
    STDN Default Setting: Stack 3 LocalConv
     - in LocalConv, volume convolutions are sequential
       --> In next LocalConv, it receive final results (after gating) as input
     - flow convolutions only get very first flow input as input
       --> Official code is wrong!
       --> Need to modify them as volume convolution
    1 LocalConv: relu(vol_cnn()) * sigmoid(relu(flow_cnn()))

    Additional Change Note
     - We need to reshape the input from (B, T, S, S, C) to (B*T, S, S, C)
     - We utilize shareable weights among the CNNs different time step
       --> Originally, STDN utilizes individual CNN module per time step
    """
    def __init__(self, vol_in=2, flow_in=4, cnn_channels = 64, flat_size = 128, kernel_size = (3,3), num_layer = 3, nbhd_size = 7):
        super(StackedLocalConv, self).__init__()
        self.convs = nn.ModuleList()
        self.nbhd_size = nbhd_size
        self.flat_size = flat_size
        for _ in range(num_layer):
            self.convs.append(LocalConv(vol_in, flow_in, cnn_channels, kernel_size))
            vol_in = flow_in = cnn_channels
        self.fc = nn.Linear(cnn_channels * (nbhd_size ** 2), flat_size)

    def make_local_patches(self, x):
        """
        Get input B, H, W, C and neighbor size S
        Unfold them as B, S, S, C, H*W and merge them as (B*H*W, S, S, C)
        """
        if x.dim() > 4:
            B, H, W, S, S, H = x.size()
            x_unfold = x.view(-1, S, S, H)
            x_unfold = x_unfold.permute(0,3,1,2).contiguous()
            return x_unfold
        B, H, W, C = x.size()
        x = x.permute(0,3,1,2).contiguous() # B, C, H, W
        S = self.nbhd_size
        padding_size = S // 2
        # B, C, S, S, H * W
        x_unfold = F.unfold(input = x, kernel_size = (S,S), stride = (1,1), padding = (padding_size,padding_size)).reshape(B, C, S, S, -1)
        x_unfold = x_unfold.permute(0,4,1,2,3).reshape(-1, C, S, S)
        return x_unfold

    def forward(self, x_vol, x_flow):
        B, H, W, _ = x_vol.size()
        x_vol = self.make_local_patches(x_vol)
        x_flow = self.make_local_patches(x_flow)
        for conv in self.convs:
            x_vol, x_flow = conv(x_vol, x_flow)
        x_vol_flat = x_vol.reshape(B*H*W, -1)

        return torch.relu(self.fc(x_vol_flat))

class LocalConv(nn.Module):
    def __init__(self, vol_in, flow_in, out_channels = 64, kernel_size = (3,3)):
        super(LocalConv, self).__init__()
        self.conv_v = nn.Conv2d(vol_in, out_channels, kernel_size = kernel_size, padding = 'same')
        self.conv_f = nn.Conv2d(flow_in, out_channels, kernel_size = kernel_size, padding = 'same')

    def forward(self, x_vol, x_flow):
        conv_vol = torch.relu(self.conv_v(x_vol))
        conv_flow = torch.relu(self.conv_f(x_flow))
        gated_out = conv_vol * torch.sigmoid(conv_flow)
        return gated_out, conv_flow

class ext_lstm(nn.Module):
    def __init__(self, ext_size, flat_size = 128, lstm_out_size=128, is_cur=True):
        super(ext_lstm, self).__init__()
        self.lstm = nn.LSTMCell(input_size = ext_size + flat_size, hidden_size = lstm_out_size)
        self.is_cur = is_cur

    def forward(self, x_cnn, x_ext):
        B, T, _ = x_ext.size()
        x_cnn = x_cnn.reshape(B, T, -1)
        lstm_in = torch.cat([x_cnn, x_ext], dim = -1)
        hx, cx = self.lstm(lstm_in[:,0])
        hiddens = [hx]
        for i in range(1, T):
            hx, cx = self.lstm(lstm_in[:,i], (hx, cx))
            hiddens.append(hx)
        if self.is_cur:
            hiddens = hiddens[-1]
        else:
            hiddens = torch.stack(hiddens, dim = 1)
        return hiddens



class PSAM(nn.Module):
    """
    Periodically Shifted Attention Mechanism
    1. Get representative hidden states of each day as: Attn(h_{p,q}, h) = h_p
       --> Note: attention is calculated with today & previous day
       --> attention score: v.transpose * tanh(Wh * h_{p,q} + Wx * h + b)
    2. Apply additional LSTM with h_p to preserve the sequential information
    """
    def __init__(self, lstm_out_size=128, bias = False):
        super(PSAM, self).__init__()
        self.query = nn.Linear(lstm_out_size, lstm_out_size, bias = False)
        self.key = nn.Linear(lstm_out_size, lstm_out_size, bias = False)
        self.value = nn.Parameter(torch.zeros(lstm_out_size, 1))
        self.lstm = nn.LSTMCell(lstm_out_size, lstm_out_size)

    def forward(self, cur_hidden, hist_hiddens):
        query_vector = self.query(cur_hidden)
        lstm_inputs = []
        for hist_hidden in hist_hiddens:
            key_vectors = self.key(hist_hidden)
            energy = torch.tanh(query_vector.unsqueeze(dim = 1) + key_vectors)
            score = torch.matmul(energy, self.value)
            att_out = torch.sum(hist_hidden * torch.softmax(score, dim = 1), dim = 1)
            lstm_inputs.append(att_out)

        hx, cx = torch.zeros_like(lstm_inputs[0]), torch.zeros_like(lstm_inputs[0])
        for lstm_input in lstm_inputs:
            hx, cx = self.lstm(lstm_input, (hx, cx))
        return hx




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
    parser.add_argument('--ext_size', default = 160)
    parser.add_argument('--att_lstm_num', default = 3)
    parser.add_argument('--lstm_out_size', default = 128)
    args = parser.parse_args()
    vol_default = torch.zeros(8,20,10,2)
    flow_default = torch.zeros(8,20,10,4)
    att_cnn_x = [vol_default for _ in range(9)]
    att_flow = [flow_default for _ in range(9)]
    att_x = [torch.zeros(8*20*10, 3, 160) for _ in range(3)]
    cnn_x = [vol_default for _ in range(7)]
    flow = [flow_default for _ in range(7)]
    x = torch.zeros(8*20*10,7,160)
    y = torch.zeros(8,20,10,2)
    model = STDN(args)
    print(model(att_cnn_x, att_flow, att_x, cnn_x, flow, x).shape)

    
