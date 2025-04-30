import numpy as np
import pandas as pd
from datetime import datetime, timedelta
# from sklearn.preprocessing import MinMaxScaler

# y=10 lat, x=20 lon grid 생성 함수
def get_grid_index(lat, lon, lat_min=None, lat_max=None, lon_min=None, lon_max=None, grid_size=(10, 20)):
    lat_step = (lat_max - lat_min) / grid_size[0]
    lon_step = (lon_max - lon_min) / grid_size[1]
    lat_idx = int((lat - lat_min) / lat_step)
    lon_idx = int((lon - lon_min) / lon_step)
    lat_idx = min(max(lat_idx, 0), grid_size[0] - 1)
    lon_idx = min(max(lon_idx, 0), grid_size[1] - 1)
    return lat_idx, lon_idx

def main(args):
    df = pd.read_csv(args.input_csv)
    # # 데이터 로드 (CSV 대신 예제 DataFrame 사용)
    # df = pd.DataFrame([
    #     ["20240531055303", "20240601000004", 9009721, 37.5135, 126.9432, 9009660, 37.5032, 126.9607, 1],
    #     ["20240531043808", "20240601000006", 9010307, 37.5121, 126.9441, 9009701, 37.5085, 126.9467, 1],
    #     ["20240531053102", "20240601000016", 9009674, 37.5075, 126.9613, 9009660, 37.5032, 126.9607, 1],
    #     # 추가 데이터...
    # ], columns=["운행출발일시", "승차일시", "승차정류장ID", "승차정류장위도", "승차정류장경도", "하차정류장ID", "하차정류장위도", "하차정류장경도", "이용객수"])

    # 하차일시가 NULL인 데이터 제거
    df.dropna(subset=["하차일시"], inplace=True)
    # 날짜 변환
    df["승차일시"] = pd.to_datetime(df["승차일시"], format="%Y%m%d%H%M%S")

    lat_min = min(df["승차정류장위도"].min(), df["하차정류장위도"].min())
    lat_max = max(df["승차정류장위도"].max(), df["하차정류장위도"].max())
    lon_min = min(df["승차정류장경도"].min(), df["하차정류장경도"].min())
    lon_max = max(df["승차정류장경도"].max(), df["하차정류장경도"].max())
    print("lon x,lat y:", lon_min, lon_max, lat_min, lat_max)

    # 격자 좌표 변환
    df["승차_grid"] = df.apply(lambda row: get_grid_index(row["승차정류장위도"], row["승차정류장경도"], 
                            lat_min=lat_min, lat_max=lat_max , lon_min=lon_min, lon_max=lon_max, 
                            grid_size=(args.grid_y, args.grid_x)), axis=1)
    df["하차_grid"] = df.apply(lambda row: get_grid_index(row["하차정류장위도"], row["하차정류장경도"], 
                            lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
                            grid_size=(args.grid_y, args.grid_x)), axis=1)
    print(df.head())
    # df.to_csv("data/dongjak_OD_station_lat_lon_grid.csv")
    # 시간 축 정의
    start_time = df["승차일시"].min().replace(minute=0,second=0)
    end_time = df["승차일시"].max()
    time_bins = pd.date_range(start=start_time, end=end_time, freq=args.time_interval)  # 30분 단위

    # Flow 데이터 초기화 (2 방향, 시간 개수, 출발지 10x20, 도착지 10x20)
    time_steps = len(time_bins) - 1
    flow_data = np.zeros((2, time_steps, 10, 20, 10, 20))
    # Volume 데이터 초기화 (시간 개수, 10, 20, 2)
    volume_data = np.zeros((time_steps, 10, 20, 2))

    # 시간별 Flow, Volume 계산
    for t in range(time_steps):
        time_start = time_bins[t]
        time_end = time_bins[t + 1]
        df_t = df[(df["승차일시"] >= time_start) & (df["승차일시"] < time_end)]

        for _, row in df_t.iterrows():
            i, j = row["승차_grid"]
            k, l = row["하차_grid"]

            # Flow 데이터 (출발 → 도착)
            flow_data[0, t, i, j, k, l] += row["이용객수"]
            flow_data[1, t, k, l, i, j] += row["이용객수"]  # 반대 방향

            # Volume 데이터 (승차량/하차량)
            volume_data[t, i, j, 0] += row["이용객수"]  # 승차량
            volume_data[t, k, l, 1] += row["이용객수"]  # 하차량

    # Min-Max Normalization
    # flow_scaler = MinMaxScaler()
    # volume_scaler = MinMaxScaler()
    # flow_data = flow_scaler.fit_transform(flow_data.reshape(-1, 1)).reshape(flow_data.shape)
    # volume_data = volume_scaler.fit_transform(volume_data.reshape(-1, 1)).reshape(volume_data.shape)

    # 훈련 / 테스트 데이터 분리
    train_size = int(time_steps * 0.66)  # 80% train, 20% test
    flow_train, flow_test = flow_data[:, :train_size], flow_data[:, train_size:]
    volume_train, volume_test = volume_data[:train_size], volume_data[train_size:]

    time_train = np.array([t.strftime("%Y-%m-%d %H:%M:%S") for t in time_bins[:-1][:train_size]])
    time_test = np.array([t.strftime("%Y-%m-%d %H:%M:%S") for t in time_bins[:-1][train_size:]])

    # 데이터 저장 (npz 포맷)
    np.savez("data/ciel/djOD_flow_train.npz", flow=flow_train, time=time_train)
    np.savez("data/ciel/djOD_flow_test.npz", flow=flow_test, time=time_test)
    np.savez("data/ciel/djOD_volume_train.npz", volume=volume_train, time=time_train)
    np.savez("data/ciel/djOD_volume_test.npz", volume=volume_test, time=time_test)

    print("STDN 모델 인풋 데이터 생성 완료!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="STDN 모델 학습용 데이터 생성")
    parser.add_argument("--input_csv", type=str, default="data/dongjak_OD_station_lat_lon.csv", 
                        help="입력 데이터 CSV 파일 경로")
    parser.add_argument("--output_dir", type=str, default="data/ciel/", help="생성된 데이터 저장 경로")
    parser.add_argument("--grid_x", type=int, default=20, help="longitude grid 개수")
    parser.add_argument("--grid_y", type=int, default=10, help="latitude 방향 grid 개수")
    parser.add_argument("--time_interval", type=str, default="30min", help="시간 간격 설정 (예: '30min', '1H')")
    # 훈련 / 테스트 데이터 비율
    parser.add_argument("--train_ratio", type=float, default=0.66, 
                        help="훈련 데이터 비율 (예: 0.66이면 66%를 학습 데이터로 사용)")
    args = parser.parse_args()
    main(args)