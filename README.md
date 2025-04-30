📍 STDN용 동작구 마을버스 OD 데이터 처리 가이드

본 저장소는 STDN (Spatio-Temporal Dynamic Network) 모델에 적용 가능한 동작구 마을버스 OD 데이터를 전처리하고 변환하는 전체 파이프라인을 제공합니다.<br>
데이터는 dongjak_OD.xlsx (강준구 이사님 제공) 파일을 기반으로 합니다.

⸻

🔸 데이터 구성<br>

1. 원시 데이터<br>
	•	dongjak_OD.xlsx: 마을버스 OD 데이터 원본<br>
	•	→ 탭1, 탭2를 통합하여 dongjak_OD_dateOder.csv 생성<br>
✅ dongjak_OD_dateOder.csv 예시<br>
```카드번호,운행출발일시,트랜잭션ID,교통수단CD,환승횟수,버스노선ID,교통사업자ID,차량ID,사용자구분코드,승차일시,승차정류장ID,하차일시,하차정류장ID,이용객수_다인승```
```U0000080222177,20240531055303,23,105,1,11110602,111520020,111753597,1,20240601000004,9009721,20240601001538,9009660,1```

⸻

2. 필수 보조 데이터<br>
	•	dongjak_stationID_lat_lon.csv: STDN 모델에 필요한 정류장 위치 정보<br>
	•	dongjak_OD_stack1.csv / dongjak_OD_cur1.csv: 30분 단위 OD 데이터<br>

⸻

🔧 데이터 처리 파이프라인<br>

1. OD + 위치 데이터 통합<br>
  OD와 정류장 위치 데이터를 통합하여 STDN 입력용 형식 생성<br>
  
  •	실행 파일:<br>
dongjakOD_concat_stationID.ipynb 또는 dongjakOD_concat_stationID.py<br>
  •	출력 파일:<br>
dongjak_OD_station_lat_lon.csv<br>

✅ 출력 예시<br>

```운행출발일시,승차일시,승차정류장ID,승차정류장번호,승차정류장명,승차정류장위도,승차정류장경도,승차정류장방위,```
```하차일시,하차정류장ID,하차정류장번호,하차정류장명,하차정류장위도,하차정류장경도,하차정류장방위,이용객수,환승횟수```
```20240531055303,20240601000004,9009721,119900126.0,노량진역,37.5135038929,126.9432478962,87.0,```
```20240601001538,9009660,119900092.0,은로초등학교,37.5032573835,126.9607814881,198.0,1,1```

⚠️ 주의: dongjak_OD_station_lat_lon.csv 파일이 반드시 생성되어야만 STDN 학습/추론용 데이터 생성이 가능합니다.<br>

⸻

2. STDN 모델 입력 데이터 생성<br>

최소 학습 단위 생성 스크립트<br>
	•	실행 파일: generate_dj_OD2stdnData.py<br>
	•	최소 484개 샘플 필요 (483개 과거 + 1개 현재)<br>
	•	샘플 1개 = 30분간 OD 데이터<br>

유닛 데이터 변환용 스크립트<br>
	•	실행 파일: generate_unit_dj_OD2stdnData.py<br>
	•	입력 파일: dongjak_OD_station_lat_lon_grid.csv<br>
(정류장 좌표가 grid 기반으로 정렬된 형태)<br>

⸻

✅ 실행 순서 요약<br>

  1.	OD 데이터 통합<br>
dongjakOD_concat_stationID.ipynb 실행 → dongjak_OD_station_lat_lon.csv 생성<br>
	2.	STDN 데이터 생성<br>
	•	일반 학습 데이터: generate_dj_OD2stdnData.py<br>
	•	단일 유닛 기준 데이터: generate_unit_dj_OD2stdnData.py<br>
