## data 
Test Backup DB 에 접속해서 데이터를 가져와야 함<br>
ip: 211.115.111.56 (bak.ciel.co.kr)<br>
- 데이터가 있는 경로<br>
  F:\RND_DATA\Ahn_Data\STDN_API<br>

## api
server_infer_demand.py : 수요예측 추론 api<br>
server_train_demand.py : 수요예측 학습 api<br>
- 실행 : (예시) fastapi dev server_infer_demand.py

## STDN용 동작구 마을버스 OD 데이터 처리 가이드

본 저장소는 STDN (Spatio-Temporal Dynamic Network) 모델에 적용 가능한 동작구 마을버스 OD 데이터를 전처리하고 변환하는 전체 파이프라인을 제공함.<br>

- [참고] 강준구 이사님이 주신 마을버스 데이터(dongjak_OD.xlsx) 기반으로 만들어짐.<br>  
    → 위 마을버스 데이터 탭1,2을 통합하여 “dongjak_OD_dateOder.csv”으로 만들어짐<br>
    “dongjak_OD_dateOder.csv” 예시 :<br>
카드번호,운행출발일시,트랜잭션ID,교통수단CD,환승횟수,버스노선ID,교통사업자ID,차량ID,사용자구분코드,승차일시,승차정류장ID,하차일시,하차정류장ID,이용객수_다인승<br>
0000080222177,20240531055303,23,105,1,11110602,111520020,111753597,1,20240601000004,9009721,20240601001538,9009660,1<br>

⸻

- 필수 데이터<br>
	• 동작구 OD 데이터: “dongjak_OD_dateOder.csv”<br>
	• STDN데이터는 정류장의 위치 정보 데이터: "dongjak_stationID_lat_lon.csv”<br>
<br>
- unit inference 하기 위해 필요 데이터 <br>
	•	dongjak_OD_stack1.csv / dongjak_OD_cur1.csv: 30분 단위 OD 데이터<br>

⸻

데이터 처리 파이프라인<br>

1. OD + 위치 데이터 통합<br>
  OD와 정류장 위치 데이터를 통합하여 STDN 입력용 형식 생성<br>
  
  •	실행 파일:<br>
dongjakOD_concat_stationID.ipynb 또는 dongjakOD_concat_stationID.py<br>
  •	출력 파일:<br>
dongjak_OD_station_lat_lon.csv 출력 예시<br>
<br>
운행출발일시,승차일시,승차정류장ID,승차정류장번호,승차정류장명,승차정류장위도,승차정류장경도,승차정류장방위,<br>
하차일시,하차정류장ID,하차정류장번호,하차정류장명,하차정류장위도,하차정류장경도,하차정류장방위,이용객수,환승횟수<br>
20240531055303,20240601000004,9009721,119900126.0,노량진역,37.5135038929,126.9432478962,87.0,<br>
20240601001538,9009660,119900092.0,은로초등학교,37.5032573835,126.9607814881,198.0,1,1<br>

주의: dongjak_OD_station_lat_lon.csv 파일이 반드시 생성되어야만 STDN 학습/추론용 데이터 생성이 가능.<br>

⸻

2. STDN 모델 입력 데이터 생성<br>

inference 가능한 데이터 생성 스크립트<br>
	•	실행 파일: generate_dj_OD2stdnData.py<br>
	•	최소 484개 샘플 필요 (483개 과거 + 1개 현재)<br>
	•	샘플 1개 = 30분간 OD 데이터<br>

generate_unit_dj_OD2stdnData.py 에서 unit data 생성 스크립트 (unit data: 추론 가능한 최소단위 데이터)<br>
	•	실행 파일: generate_unit_dj_OD2stdnData.py<br>
	•	입력 파일: dongjak_OD_station_lat_lon_grid.csv<br>
(정류장 좌표가 grid 기반으로 정렬된 형태) “dongjak_OD_station_lat_lon_grid.csv” 형태로 데이터 프레임이 되어 있다는 가정하에 unit data 생성<br>

⸻

