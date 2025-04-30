import pandas as pd
import numpy as np

df_od = pd.read_excel("publicDB/dongjak_OD_dateOder.csv")
df_staion = pd.read_csv("publicDB/dongjak_stationID_lat_lon.csv")

# 날짜 형식으로 바꾸기 
df_od["승차일시"] = pd.to_datetime(df_od["승차일시"], format="%Y%m%d%H%M%S", errors="coerce")
df = df_od.sort_values(by="승차일시").reset_index(drop=True)