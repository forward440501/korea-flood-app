# 자료 출처와 이용 조건

- **경계** — [vuski/admdongkor, ver20261001](https://github.com/vuski/admdongkor/tree/master/ver20261001), `HangJeongDong_ver20261001.geojson`. 본 데이터는 통계청 통계지리정보서비스(SGIS, https://sgis.kostat.go.kr)에서 공공누리 제1유형으로 개방한 행정동 경계를 가공한 것이며(가공: vuski/admdongkor, https://github.com/vuski/admdongkor), CC BY 4.0으로 배포됩니다. 연동판에서는 시군구 코드 `sgg`로 합치고, EPSG:5179의 200m 경계 마스크로 추가 가공했습니다. [경계 데이터 라이선스](https://github.com/vuski/admdongkor/blob/master/LICENSE-DATA)
- **고도** — [Copernicus DEM GLO-30 Public](https://registry.opendata.aws/copernicus-dem/), 2021 AWS 배포판의 타일 35개를 취합했습니다. EPSG:5179로 평균 재표본화하고, 행정 경계 안에서 200m 표본을 1km 격자로 집계했습니다. [자료 이용 조건](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM)
  
  Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
  
  타일명은 `artifacts_national/manifest.json`의 `dem_tile_names`. [타일 목록](https://copernicus-dem-30m.s3.amazonaws.com/tileList.txt). 파일 URL은 `https://copernicus-dem-30m.s3.amazonaws.com/<타일명>/<타일명>.tif`.
- **건축면적** — Pesaresi, M. & Politis, P. (2023), [GHS-BUILT-S R2023A](https://human-settlement.emergency.copernicus.eu/ghs_buS2023.php), European Commission Joint Research Centre, [doi:10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA](https://doi.org/10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA). CC BY 4.0, E2020 1km를 평균 재투영하고 면적 비율로 환산했습니다. 건축면적은 도로를 포함한 전체 불투수면의 비율이 아닙니다. [E2020 1km 원자료](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_1000/V1-0/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_1000_V1_0.zip)
- **기상** — 기상청 API허브 [AWS 시간통계 3.1.3 강수](https://apihub.kma.go.kr/apiList.do?seqApi=2&seqApiSub=239), [관측 지점 좌표](https://apihub.kma.go.kr/apiList.do?seqApi=2&seqApiSub=317), [기상특보 발표기준](https://www.weather.go.kr/w/forecast/guide/standard.do). 과거 시연은 기존 연구의 기상청 ASOS 자료에서 계산했으며 [인천 ASOS 관측지점 좌표](https://data.kma.go.kr/tmeta/stn/selectStnDetail.do?isSelectStn=Y&pgmNo=82&stdStnNo=112)를 사용했습니다.
- **침수 기록** — [행정안전부 침수흔적도](https://safetydata.go.kr/disaster-data/view?dataSn=108)를 가공한 프로젝트의 기존 9B 학습표와 검증표(3개 도시). 침수 원자료를 전국으로 새로 배포하지 않습니다.
- **지도 배경** — CARTO, OpenStreetMap contributors. 지도 바탕의 표준 출처 표기를 보존합니다.
