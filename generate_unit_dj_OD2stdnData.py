import numpy as np
import pandas as pd

def get_grid_index(lat, lon, lat_min=None, lat_max=None, lon_min=None, lon_max=None, grid_size=(10, 20)):
    """
    y=10 lat, x=20 lon grid 생성 함수
    """
    lat_step = (lat_max - lat_min) / grid_size[0]
    lon_step = (lon_max - lon_min) / grid_size[1]
    lat_idx = int((lat - lat_min) / lat_step)
    lon_idx = int((lon - lon_min) / lon_step)
    lat_idx = min(max(lat_idx, 0), grid_size[0] - 1)
    lon_idx = min(max(lon_idx, 0), grid_size[1] - 1)
    return lat_idx, lon_idx

def map_station_info(df_cur, df_station):
    """
    승차정류장ID와 하차정류장ID를 기준으로 df_station에서 정류장 상세 정보(번호, 명, 위경도, 방위)를 병합
    """
    results=[]
    for i, (pickID, dropID) in enumerate(zip(df_cur["승차정류장ID"], df_cur["하차정류장ID"])):
        try:
            pick_station_row = df_station[df_station["ID"] == int(pickID)]
        except ValueError:
            pick_station_row = pd.DataFrame()  
            pickID=None
        
        try:
            drop_station_row = df_station[df_station["ID"] == int(dropID)]
        except ValueError:
            drop_station_row = pd.DataFrame()  
            dropID=None
        
        od_row = df_cur.iloc[i]
        # 기본값 설정 (정류장 정보가 없을 경우 대비)
        pick_num = pick_name = pick_lat = pick_lon = pick_azimuth = None
        drop_num = drop_name = drop_lat = drop_lon = drop_azimuth = None 
        # bus_departure_time = pick_time = None
        # drop_time = None
        if not pick_station_row.empty:
            pick_num = int(pick_station_row.iloc[0]["정류장번호"])
            pick_name = pick_station_row.iloc[0]["정류장명"]
            pick_lat = pick_station_row.iloc[0]["조정위도"]
            pick_lon = pick_station_row.iloc[0]["조정경도"]
            pick_azimuth = int(pick_station_row.iloc[0]["방위"])
            
        if not drop_station_row.empty:
            drop_num = int(drop_station_row.iloc[0]["정류장번호"])
            drop_name = drop_station_row.iloc[0]["정류장명"]
            drop_lat = drop_station_row.iloc[0]["조정위도"]
            drop_lon = drop_station_row.iloc[0]["조정경도"]
            drop_azimuth = int(drop_station_row.iloc[0]["방위"])
        
        results.append({"운행출발일시": od_row["운행출발일시"], 
                        "승차일시": od_row["승차일시"], 
                        "승차정류장ID": pickID,
                        "승차정류장번호": pick_num, 
                        "승차정류장명": pick_name, 
                        "승차정류장위도": pick_lat, 
                        "승차정류장경도": pick_lon, 
                        "승차정류장방위": pick_azimuth,
                        "하차일시": od_row["하차일시"] if od_row["하차일시"] != '~' else None, 
                        "하차정류장ID": dropID, 
                        "하차정류장번호": drop_num, 
                        "하차정류장명": drop_name, 
                        "하차정류장위도": drop_lat, 
                        "하차정류장경도": drop_lon, 
                        "하차정류장방위": drop_azimuth,
                        "이용객수" : od_row["이용객수_다인승"], 
                        "환승횟수": od_row["환승횟수"]
        })

    result_df = pd.DataFrame(results)
    print(result_df.head())    
    return result_df
    

def main(args):
    cur_df = pd.read_csv(args.cur_csv)
    stack_df = pd.read_csv(args.stack_csv)  
    station_df = pd.read_csv(args.station_csv)
    
    # # 데이터 로드 (CSV 대신 예제 DataFrame 사용)
    # df = pd.DataFrame([
    #     ["20240531055303", "20240601000004", 9009721, 37.5135, 126.9432, 9009660, 37.5032, 126.9607, 1],
    #     ["20240531043808", "20240601000006", 9010307, 37.5121, 126.9441, 9009701, 37.5085, 126.9467, 1],
    #     ["20240531053102", "20240601000016", 9009674, 37.5075, 126.9613, 9009660, 37.5032, 126.9607, 1],
    #     # 추가 데이터...
    # ], columns=["운행출발일시", "승차일시", "승차정류장ID", "승차정류장위도", "승차정류장경도", "하차정류장ID", "하차정류장위도", "하차정류장경도", "이용객수"])
    
    total_cur_df = map_station_info(cur_df, station_df)

    # stack_df + total_cur_df
    try:
        # 컬럼 이름과 순서가 같은지 검사
        if list(stack_df.columns) != list(total_cur_df.columns):
            raise ValueError("컬럼 이름 또는 순서가 다릅니다.")
        
        combined_df = pd.concat([stack_df, total_cur_df], axis=0, ignore_index=True)

    except ValueError as e:
        print(e)
        print("▶ stack_df columns:", list(stack_df.columns))
        print("▶ total_cur_df columns:", list(total_cur_df.columns))
        combined_df = None  # 오류 발생 시 None 처리

    # 하차일시가 NULL인 데이터 제거
    combined_df.dropna(subset=["하차일시"], inplace=True)
    # 승차정류장위도/경도, 하차정류장위도/경도: NULL인 데이터 제거
    combined_df.dropna(subset=["승차정류장위도"], inplace=True)
    combined_df.dropna(subset=["하차정류장위도"], inplace=True)
    combined_df.dropna(subset=["승차정류장경도"], inplace=True)
    combined_df.dropna(subset=["하차정류장경도"], inplace=True)
    # 날짜 변환
    # combined_df["승차일시"] = pd.to_datetime(combined_df["승차일시"], format="%Y%m%d%H%M%S")

    lat_min = min(combined_df["승차정류장위도"].min(), combined_df["하차정류장위도"].min())
    lat_max = max(combined_df["승차정류장위도"].max(), combined_df["하차정류장위도"].max())
    lon_min = min(combined_df["승차정류장경도"].min(), combined_df["하차정류장경도"].min())
    lon_max = max(combined_df["승차정류장경도"].max(), combined_df["하차정류장경도"].max())
    print("lon x,lat y:", lon_min, lon_max, lat_min, lat_max)

    # 격자 좌표 변환
    combined_df["승차_grid"] = combined_df.apply(lambda row: get_grid_index(row["승차정류장위도"], row["승차정류장경도"], 
                            lat_min=lat_min, lat_max=lat_max , lon_min=lon_min, lon_max=lon_max, 
                            grid_size=(args.grid_y, args.grid_x)), axis=1)
    combined_df["하차_grid"] = combined_df.apply(lambda row: get_grid_index(row["하차정류장위도"], row["하차정류장경도"], 
                            lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
                            grid_size=(args.grid_y, args.grid_x)), axis=1)
    # print(combined_df.head())
    # df.to_csv("data/dongjak_OD_station_lat_lon_grid.csv")
    
    ## 30분씩 df 생성  
    # 시간 축 정의
    # 1. 승차일시를 datetime으로 변환
    combined_df["승차일시"] = pd.to_datetime(combined_df["승차일시"], format="%Y%m%d%H%M%S", errors="coerce")
    # 2. 시작/끝 시간 정의 (초, 분 제거해서 깔끔하게 맞춤)
    start_time = combined_df["승차일시"].min().floor(args.time_interval)
    end_time = combined_df["승차일시"].max().ceil(args.time_interval) #+ pd.Timedelta(minutes=30)
    # 3. 30분 단위 타임라인 생성
    time_bins = pd.date_range(start=start_time, end=end_time, freq=args.time_interval)  # 30분 단위
    
    # Flow 데이터 초기화 (2 방향, 시간 개수, 출발지 10x20, 도착지 10x20)
    time_steps = len(time_bins) -1

    flow_data = np.zeros((2, len(time_bins), 10, 20, 10, 20))
    # Volume 데이터 초기화 (시간 개수, 10, 20, 2)
    volume_data = np.zeros((len(time_bins), 10, 20, 2))

    # 시간별 Flow, Volume 계산
    for t in range(time_steps):
        time_start = time_bins[t]
        time_end = time_bins[t + 1]
        df_t = combined_df[(combined_df["승차일시"] >= time_start) & (combined_df["승차일시"] < time_end)]

        for _, row in df_t.iterrows():
            i, j = row["승차_grid"]
            k, l = row["하차_grid"]

            # Flow 데이터 (출발 → 도착)
            flow_data[0, t, i, j, k, l] += row["이용객수"]
            flow_data[1, t, k, l, i, j] += row["이용객수"]  # 반대 방향

            # Volume 데이터 (승차량/하차량)
            volume_data[t, i, j, 0] += row["이용객수"]  # 승차량
            volume_data[t, k, l, 1] += row["이용객수"]  # 하차량

    
    # train_size = int(time_steps) 
    print('----flow_data.shape:',flow_data.shape, ', volume_data.shape:',volume_data.shape)

    time_test = np.array([t.strftime("%Y-%m-%d %H:%M:%S") for t in time_bins])

    # 데이터 저장 (npz 포맷)
    np.savez("data/ciel/djOD_flow_cur1.npz", flow=flow_data, time=time_test)
    np.savez("data/ciel/djOD_volume_cur1.npz", volume=volume_data, time=time_test)

    print("STDN 모델 인풋 데이터 생성 완료!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="STDN 모델 학습용 데이터 생성")
    parser.add_argument("--cur_csv", type=str, default="data/dongjak_OD_cur1.csv", 
                        help="30min 현재 데이터 CSV 파일 경로")
    parser.add_argument("--stack_csv", type=str, default="data/dongjak_OD_station_stack1.csv", 
                        help="stacked 30min 과거 데이터 CSV 파일 경로")    
    parser.add_argument("--station_csv", type=str, default="data/dongjak_stationID_lat_lon.csv", 
                        help="stationID_lat_lon 정보 담긴 CSV 파일 경로")
    parser.add_argument("--output_dir", type=str, default="data/ciel/", help="생성된 데이터 저장 경로")
    parser.add_argument("--grid_x", type=int, default=20, help="longitude grid 개수")
    parser.add_argument("--grid_y", type=int, default=10, help="latitude 방향 grid 개수")
    parser.add_argument("--time_interval", type=str, default="30min", help="시간 간격 설정 (예: '30min', '1H')")
    # 훈련 / 테스트 데이터 비율
    # parser.add_argument("--train_ratio", type=float, default=0.66, 
    #                     help="훈련 데이터 비율 (예: 0.66이면 66%를 학습 데이터로 사용)")
    args = parser.parse_args()
    main(args)