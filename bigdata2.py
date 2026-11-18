import datetime
import json
import os
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from pyproj import Transformer
from streamlit_folium import st_folium

from flood_live import (
    load_archive, load_yearround_artifacts, locate, map_raster, score_snapshot,
)
from weather import KST, STATIONS

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="주요 연안/도심 지역 홍수 취약도 및 강수 비교 분석", layout="wide"
)

# 💡 공공데이터포털 발급 일반 인증키
#    코드에 직접 적지 않고, 배포 사이트의 Secrets(또는 .streamlit/secrets.toml)에서 읽습니다.
def _data_go_kr_key():
  key = os.environ.get("DATA_GO_KR_KEY", "").strip()
  if not key:
    try:
      key = str(st.secrets.get("DATA_GO_KR_KEY", "")).strip()
    except Exception:
      key = ""
  return key


API_KEY = _data_go_kr_key()

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"      # 기존 3도시 100m 모델·격자·과거 ASOS
EVIDENCE = BASE_DIR / "evidence"        # 연구 보고서의 검증표·그림

# 2. 발표용 대상 지역 기본 데이터 (강릉, 인천, 거제)
region_data = {
    "지역명": ["강릉시", "인천 중구", "거제시"],
    "stn_id": [105, 112, 294],  # 기상청 ASOS 지점 코드
    "lat": [37.7519, 37.4692, 34.8806],
    "lon": [128.8760, 126.6322, 128.6211],
    "zoom": [11, 11, 11],  # 각 지역별 줌 레벨
    "취약도등급": ["상 (High)", "상 (High)", "중/상 (Med-High)"],
    "주요 취약 원인": [
        "해안가 만조 수위 결합 & 산지 유출수 도심 하천(남대천) 병목",
        "서해안 만조(사리) 감조구역 내수 배수 지연 & 불투수면 비율 높음",
        "산지 급경사 지표 유출수 단시간 해안/산업단지 저지대 집중",
    ],
}
df_regions = pd.DataFrame(region_data)
REGION_TO_CITY = {"강릉시": "강릉", "인천 중구": "인천", "거제시": "거제"}
CITY_LABEL = {"강릉": "강릉", "인천": "인천(강화군 제외)", "거제": "거제"}

# 점수 종류: 7~10월 기상 관측이 갖춰지면 기상 반영 점수, 아니면 지형·피복 기본 점수
KIND_NAME = {"weather": "기상 반영 점수", "baseline": "기본 점수 (지형·토지피복)"}
KIND_LEGEND = {
    "weather": "linear-gradient(90deg,#2c509e,#30aca4,#faa540)",
    "baseline": "linear-gradient(90deg,#e1eaf1,#6f90a8,#26435c)",
}

# 모델 별도 평가 기간의 실제 호우 사례
EVENT_PRESETS = {
    "직접 선택": None,
    "2022-08-08 21시 · 인천 호우": (datetime.date(2022, 8, 8), 21),
    "2022-08-16 22시 · 강릉 호우": (datetime.date(2022, 8, 16), 22),
    "2021-08-23 20시 · 거제 호우": (datetime.date(2021, 8, 23), 20),
}


# 🗺️ GeoJSON 행정경계 데이터 로드 함수
@st.cache_data
def load_geojson_data():
  geojson_path = BASE_DIR / "sig.geojson"
  if os.path.exists(geojson_path):
    try:
      with open(geojson_path, "r", encoding="utf-8") as f:
        return json.load(f)
    except Exception:
      try:
        with open(geojson_path, "r", encoding="cp949") as f:
          return json.load(f)
      except Exception:
        return None
  return None


geojson_data = load_geojson_data()


# 4. 🌐 기상청 ASOS API 단일 지점 조회 함수 (일강수량)
#    조회에 실패하면 가상 데이터를 만들지 않고 빈 표를 돌려줍니다.
@st.cache_data(ttl=3600)
def fetch_single_stn_data(stn_id, year, month):
  start_date = f"{year}{month:02d}01"
  if month in [1, 3, 5, 7, 8, 10, 12]:
    end_day = 31
  elif month in [4, 6, 9, 11]:
    end_day = 30
  else:
    end_day = 29 if year % 4 == 0 else 28

  today = datetime.date.today()
  if year == today.year and month == today.month:
    end_day = (today - datetime.timedelta(days=1)).day

  end_date = f"{year}{month:02d}{end_day:02d}"

  url = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
  params = {
      "serviceKey": API_KEY,
      "pageNo": "1",
      "numOfRows": "999",
      "dataType": "JSON",
      "dataCd": "ASOS",
      "dateCd": "HR",
      "startDt": start_date,
      "startHh": "00",
      "endDt": end_date,
      "endHh": "23",
      "stnIds": str(stn_id),
  }

  try:
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    items = data["response"]["body"]["items"]["item"]
    daily_rain = {}

    for item in items:
      tm_str = item["tm"]
      date_part = datetime.datetime.strptime(tm_str[:10], "%Y-%m-%d").date()
      rn = float(item["rn"]) if item.get("rn") and item["rn"] != "" else 0.0
      rn = max(0.0, rn)
      daily_rain[date_part] = daily_rain.get(date_part, 0.0) + rn

    records = [
        {"날짜": d, "일강수량_mm": round(r, 1)} for d, r in sorted(daily_rain.items())
    ]
    return pd.DataFrame(records), True

  except Exception:
    return pd.DataFrame(columns=["날짜", "일강수량_mm"]), False


# 5. 3개 지역 데이터 병합 함수
def fetch_all_regions_data(year, month):
  frames, failed = [], []
  for idx, row in df_regions.iterrows():
    df_stn, ok = fetch_single_stn_data(row["stn_id"], year, month)
    df_stn = df_stn.copy()
    df_stn["지역명"] = row["지역명"]
    frames.append(df_stn)
    if not ok:
      failed.append(row["지역명"])
  return pd.concat(frames, ignore_index=True), failed


# ==========================================
# 🧪 3도시 100m 침수 연구 모델 (연중 표시판)
# ==========================================
@st.cache_resource(show_spinner=False)
def model_bundle():
  return load_yearround_artifacts(ARTIFACTS)


@st.cache_resource(show_spinner=False)
def asos_archive():
  """연구에 사용한 실제 ASOS 시간 관측 (2006-09 ~ 2026-08)."""
  return load_archive(ARTIFACTS, model_bundle())


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hourly_obs(station, start_kst, end_kst):
  """공공데이터포털 ASOS 시간자료 → 모델 입력 형식. 강수 공백은 빈 값으로 둡니다."""
  params = {
      "serviceKey": API_KEY,
      "pageNo": "1",
      "numOfRows": "999",
      "dataType": "JSON",
      "dataCd": "ASOS",
      "dateCd": "HR",
      "startDt": start_kst.strftime("%Y%m%d"),
      "startHh": start_kst.strftime("%H"),
      "endDt": end_kst.strftime("%Y%m%d"),
      "endHh": end_kst.strftime("%H"),
      "stnIds": str(station),
  }
  url = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
  data = requests.get(url, params=params, timeout=15).json()
  items = data["response"]["body"]["items"]["item"]
  if isinstance(items, dict):
    items = [items]
  frame = pd.DataFrame([
      {
          "station": int(it.get("stnId", station)),
          "time_kst": it["tm"],
          "rain_mm": it.get("rn"),
          "temp_c": it.get("ta"),
          "rh_pct": it.get("hm"),
          "slp_hpa": it.get("ps"),  # 해면기압
          "wind_m_s": it.get("ws"),
      }
      for it in items
  ])
  for c in ["rain_mm", "temp_c", "rh_pct", "slp_hpa", "wind_m_s"]:
    frame[c] = pd.to_numeric(frame[c].replace("", np.nan), errors="coerce")
  return frame


def hourly_obs(station, issue):
  """기준시각 이전 30시간 관측: 보유 과거 관측 → 공공데이터포털 순서. (자료, 출처, 오류)"""
  start, end = issue - pd.Timedelta(hours=30), issue - pd.Timedelta(hours=1)
  try:
    arc = asos_archive()
    part = arc.loc[arc.station.eq(station) & arc.time_kst.between(start, end)]
    if len(part) >= 30:
      return part, "보유 ASOS 과거 관측", None
  except Exception:
    part = pd.DataFrame()
  try:
    return fetch_hourly_obs(station, start, end), "공공데이터포털 ASOS 시간자료", None
  except Exception:
    if not part.empty:
      return part, "보유 ASOS 과거 관측(일부 시간)", None
    return None, None, "이 시각의 ASOS 시간 관측을 받지 못했습니다."


@st.cache_data(max_entries=24, show_spinner=False)
def city_snapshot(city, issue_iso):
  """선택 시각의 도시 100m 격자 점수. 기상 관측이 없어도 기본 점수를 돌려줍니다."""
  issue = pd.Timestamp(issue_iso)
  records, source, error = hourly_obs(STATIONS[city], issue)
  snap = score_snapshot(city, issue, model_bundle(), records, observation_error=error)
  return snap, source


@st.cache_data(max_entries=24, show_spinner=False)
def city_layer(city, issue_iso):
  snap, _ = city_snapshot(city, issue_iso)
  return map_raster(snap.scores, snap.kind)


@st.cache_data(show_spinner=False)
def city_bounds(city):
  g = model_bundle().base_by_city[city]
  wgs = Transformer.from_crs(5179, 4326, always_xy=True)
  lon0, lat0, lon1, lat1 = wgs.transform_bounds(
      g.x_m.min() - 50, g.y_m.min() - 50, g.x_m.max() + 50, g.y_m.max() + 50
  )
  return [[lat0, lon0], [lat1, lon1]]


def num(v):
  return None if v is None or pd.isna(v) else round(float(v), 1)


def apply_preset():
  preset = EVENT_PRESETS[st.session_state["event_preset"]]
  if preset:
    st.session_state["target_date"], st.session_state["target_hour"] = preset


# ==========================================
# 💡 사이드바 정보 (메뉴 없음 · 한 페이지 구성)
# ==========================================
st.sidebar.title("📌 시스템 정보")
st.sidebar.info(
    "본 대시보드는 기상청 ASOS 관측 데이터를 기반으로 연안 도시 침수 취약도를 비교합니다."
    " 점수는 과거 침수 기록과 지형·토지피복·기상 관측의 관계를 학습한 상대 점수이며,"
    " 침수 확률이나 공식 경보가 아닙니다."
)
st.sidebar.markdown(
    "**페이지 구성**\n\n1. 3개 도시 지도·강수 비교\n\n2. 취약도 비교 요약\n\n"
    "3. 전국 침수 참고지도 (1km)\n\n4. 연구 결과 요약"
)


# ==========================================
# 1. 3개 도시 대시보드
# ==========================================
def page_three_cities():
  st.title("🌊 해안/도심 주요 지역 기상 관측 및 침수 취약도 비교 분석 시스템")
  st.caption(
      "기상청 ASOS 시간 관측과 100m 격자 지형·토지피복을 결합해 강릉, 인천, 거제의"
      " 침수 취약도를 비교합니다."
  )
  try:
    model_bundle()
  except Exception as e:
    st.error(f"침수 연구 모델을 불러오지 못했습니다: {e}")
    return

  col1, col2 = st.columns([1, 1.1])

  with col1:
    st.subheader("📍 1. 대상 지역별 상세 지도 및 위치")
    map_area = st.container()  # 날짜·시각을 고른 뒤 이 자리에 지도를 그립니다

    st.markdown("---")
    st.markdown("### 📅 분석 대상 날짜 지정")

    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    st.session_state.setdefault("target_date", yesterday)
    st.session_state.setdefault("target_hour", 12)

    st.selectbox(
        "대표 호우 사례 바로가기",
        list(EVENT_PRESETS),
        key="event_preset",
        on_change=apply_preset,
        help="모델 별도 평가에 쓰인 실제 호우 사례입니다. 고르면 날짜와 시각이 바뀝니다.",
    )
    d_col1, d_col2 = st.columns(2)
    with d_col1:
      target_date = st.date_input(
          "조회 날짜 지정",
          min_value=datetime.date(2007, 1, 1),
          max_value=yesterday,
          key="target_date",
      )
      target_hour = st.select_slider(
          "기준 시각 (정시, 한국시각)",
          options=list(range(24)),
          format_func=lambda h: f"{h:02d}시",
          key="target_hour",
      )
    with d_col2:
      st.info(
          f"선택일: **{target_date.strftime('%Y년 %m월 %d일')} {target_hour:02d}시**\n비교"
          " 대상: **강릉, 인천, 거제 (3개 지점)**"
      )
      st.caption(
          "연중 어느 날짜든 점수를 표시합니다. 7~10월에 기상 관측이 갖춰지면 기상 반영 점수,"
          " 그 밖에는 지형·토지피복만 쓴 기본 점수입니다."
      )

    issue = pd.Timestamp(datetime.datetime.combine(target_date, datetime.time(target_hour)), tz=KST)
    issue_iso = issue.isoformat()

    results = {}
    with st.spinner("100m 격자 점수 계산 중..."):
      for city in STATIONS:
        results[city] = city_snapshot(city, issue_iso)

    with map_area:
      tabs = st.tabs(["🌊 강릉시 관측소", "🏢 인천 중구 관측소", "🏔️ 거제시 관측소"])
      for idx, tab in enumerate(tabs):
        row = df_regions.iloc[idx]
        city = REGION_TO_CITY[row["지역명"]]
        snap, source = results[city]
        with tab:
          lat_val, lon_val = float(row["lat"]), float(row["lon"])
          m = folium.Map(
              location=[lat_val, lon_val], zoom_start=int(row["zoom"]), tiles="OpenStreetMap"
          )
          rgba, bounds = city_layer(city, issue_iso)
          folium.raster_layers.ImageOverlay(
              rgba, bounds, name=KIND_NAME[snap.kind], opacity=0.91, pixelated=True,
              mercator_project=False, interactive=False,
          ).add_to(m)
          m.fit_bounds(city_bounds(city))

          if geojson_data:
            folium.GeoJson(
                geojson_data,
                name="행정경계",
                # 점수 색을 가리지 않도록 경계선만 그립니다
                style_function=lambda f: {"color": "#3182bd", "weight": 2, "fillOpacity": 0},
                tooltip=folium.GeoJsonTooltip(
                    fields=(
                        ["SIG_KOR_NM"]
                        if "SIG_KOR_NM" in geojson_data["features"][0]["properties"]
                        else list(geojson_data["features"][0]["properties"].keys())[:1]
                    ),
                    aliases=["지역명:"],
                ),
            ).add_to(m)

          color = "red" if "상" in row["취약도등급"] else "orange"
          folium.Marker(
              location=[lat_val, lon_val],
              popup=f"<b>{row['지역명']} ASOS 관측소</b>",
              tooltip=f"{row['지역명']} (취약도: {row['취약도등급']})",
              icon=folium.Icon(color=color, icon="info-sign"),
          ).add_to(m)

          mapdata = st_folium(
              m, width=480, height=320, key=f"map_{idx}", returned_objects=["last_clicked"]
          )

          st.markdown(
              f'<div style="height:9px;border-radius:8px;background:{KIND_LEGEND[snap.kind]}"></div>',
              unsafe_allow_html=True,
          )
          st.caption(f"{KIND_NAME[snap.kind]}   |   0 상대적으로 낮음 → 100 상대적으로 높음 (침수 확률 아님)")
          s_col1, s_col2 = st.columns(2)
          s_col1.metric("도시 내 점수 상위 5% 기준", f"{snap.scores.score.quantile(.95):.1f} / 100")
          clicked = (mapdata or {}).get("last_clicked")
          cell = locate(snap.scores, float(clicked["lat"]), float(clicked["lng"])) if clicked else None
          if cell is not None:
            s_col2.metric("클릭한 100m 격자 점수", f"{cell.score:.1f} / 100")
          elif clicked:
            s_col2.caption("클릭한 위치에는 분석 범위의 100m 격자가 없습니다.")
          else:
            s_col2.caption("지도를 클릭하면 해당 100m 격자의 점수가 나옵니다.")

          if snap.kind == "weather":
            r = snap.inputs
            st.caption(
                f"{source} · ASOS {STATIONS[city]} · 최근 24시간 기록 강수 합"
                f" {r.rain_known_sum_24h:.1f} mm (수치가 있는 시간 {int(r.rain_value_hours_24h)}/24)"
            )
          else:
            st.success(
                f"☀️ 기본 점수 표시 중 · {snap.reason} 비가 오기 전에도 지형·토지피복 때문에"
                " 상대적으로 취약한 곳을 확인할 수 있습니다. 같은 격자의 기본 점수는 날짜에 따라 바뀌지 않습니다."
            )

          st.warning(f"⚠️ **{row['지역명']} 침수 취약 특성:**\n\n{row['주요 취약 원인']}")

  # 6. 우측 상세 비교 분석 창
  with col2:
    st.subheader(f"📊 {target_date.year}년 {target_date.month}월 주요 3개 지역 수문/기상 비교")

    df_all, failed_regions = fetch_all_regions_data(target_date.year, target_date.month)
    if failed_regions:
      st.warning(
          f"{', '.join(failed_regions)} 관측소의 일강수량 조회에 실패했습니다."
          " 가상 값으로 채우지 않았으므로 해당 지역은 비어 있습니다."
      )
    df_target_day = df_all[df_all["날짜"] == target_date]

    cols = st.columns(3)
    for idx, row in df_regions.iterrows():
      r_name = row["지역명"]
      day_val_df = df_target_day[df_target_day["지역명"] == r_name]
      if day_val_df.empty:
        cols[idx].metric(f"{r_name}", "자료 없음", delta="위험도: -", delta_color="off")
        continue
      val = day_val_df.iloc[0]["일강수량_mm"]
      if val == 0:
        risk_status = "🟢 안전 (Low)"
      elif val < 30:
        risk_status = "🟡 주의 (Caution)"
      else:
        risk_status = "🔴 경보 (Warning)"
      cols[idx].metric(f"{r_name}", f"{val} mm", delta=f"위험도: {risk_status}", delta_color="off")

    st.markdown("---")
    st.markdown(f"#### 🌧️ {target_date.year}년 {target_date.month}월 지역별 일강수량 비교 차트")
    fig = px.bar(
        df_all,
        x="날짜",
        y="일강수량_mm",
        color="지역명",
        barmode="group",
        title=f"강릉 vs 인천 vs 거제 {target_date.month}월 일별 강수량 비교",
        labels={"일강수량_mm": "일강수량 (mm)", "날짜": "날짜", "지역명": "관측 지역"},
        color_discrete_sequence=["#1F77B4", "#FF7F0E", "#2CA02C"],
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown(f"#### 🧪 {issue.strftime('%Y-%m-%d %H시')} 기준 시간 관측과 100m 격자 점수")
    obs_rows = []
    for _, row in df_regions.iterrows():
      snap, source = results[REGION_TO_CITY[row["지역명"]]]
      inp = snap.inputs
      obs_rows.append({
          "지역명": row["지역명"],
          "점수 종류": KIND_NAME[snap.kind],
          "점수 상위 5% 기준": num(snap.scores.score.quantile(.95)),
          "최근 24시간 기록 강수 (mm)": None if inp is None else num(inp.rain_known_sum_24h),
          "최근 6시간 최대 시간강수 (mm)": None if inp is None else num(inp.rain_known_max_6h),
          "비고": snap.reason or source or "",
      })
    st.dataframe(pd.DataFrame(obs_rows), hide_index=True, width="stretch")
    st.caption(
        "도시별 대표 ASOS 한 지점의 관측을 그 도시의 모든 격자에 같이 적용합니다."
        " 기상 반영 점수와 기본 점수는 서로 다른 모델이라, 점수 종류가 바뀌면서 생긴 차이를"
        " 위험이 변했다는 뜻으로 읽으면 안 됩니다."
    )

  st.markdown("---")

  # 7. 하단 수문공학적 취약성 종합 비교 테이블
  st.subheader("📋 대상 지역 수문공학적 침수 취약도 비교 요약")
  df_summary = df_regions[["지역명", "취약도등급", "주요 취약 원인"]].copy()
  rain_list, score_list = [], []
  for r_name in df_summary["지역명"]:
    day_val_df = df_target_day[df_target_day["지역명"] == r_name]
    rain_list.append(f"{day_val_df.iloc[0]['일강수량_mm']} mm" if not day_val_df.empty else "자료 없음")
    snap, _ = results[REGION_TO_CITY[r_name]]
    score_list.append(f"{snap.scores.score.quantile(.95):.1f} / 100 ({KIND_NAME[snap.kind]})")
  df_summary[f"{target_date.strftime('%m/%d')} 관측 강수량"] = rain_list
  df_summary[f"{target_hour:02d}시 점수 상위 5% 기준"] = score_list
  st.dataframe(df_summary, hide_index=True, width="stretch")
  st.caption(
      "💡 점수는 과거 침수 시작 기록과 미관측 배경을 구분하도록 학습한 상대 점수이며,"
      " 침수 확률·침수심·공식 경보가 아닙니다. 취약도 등급과 취약 원인은 지역 특성 조사에 근거한 정성 평가입니다."
  )


# ==========================================
# 3. 전국 1km 침수 참고지도
# ==========================================
def section_national():
  st.markdown("---")
  st.subheader("🇰🇷 전국 침수 참고지도 (1km 확장)")
  try:
    from national_page import EXPLANATION, KST as NKST, _view_at, assets, demos, live_target
  except Exception as e:
    st.error(f"전국 지도 모듈을 불러오지 못했습니다: {e}")
    return
  st.caption(EXPLANATION)
  g, _ = assets()
  areas = g.areas
  c1, c2, c3 = st.columns([1.2, 1.5, 1])
  province = c1.selectbox("시도", ["전국"] + sorted({a["province"] for a in areas.values()}),
                          key="national_province")
  selectable = [a for a in areas.values() if province == "전국" or a["province"] == province]
  options = [0] + [a["id"] for a in sorted(selectable, key=lambda a: (a["province"], a["name"]))]
  area_id = c2.selectbox(
      "시군구 찾기", options,
      format_func=lambda a: f"{areas[a]['province']} {areas[a]['name']}" if a else "전국 지도",
      key="national_area")
  mode = c3.selectbox("조회 모드", ["과거 관측 시연", "실시간 관측", "날짜·시간 조회"], key="national_mode")
  rain_layer = st.toggle("지도에 강수 반영 지수 표시", value=True, key="national_rain_layer")
  target = None
  if mode == "날짜·시간 조회":
    latest = datetime.datetime.now(NKST).date()
    with st.form("national_date_form"):
      a, b, c = st.columns([1.5, 1, 1.2])
      d = a.date_input("관측 날짜", value=latest, min_value=datetime.date(1997, 1, 1),
                       max_value=latest, key="national_archive_date")
      h = b.selectbox("정시 (KST)", tuple(range(24)), format_func=lambda h: f"{h:02d}:00",
                      index=live_target(datetime.datetime.now(NKST)).hour, key="national_archive_hour")
      if c.form_submit_button("관측 조회"):
        st.session_state["national_requested"] = datetime.datetime.combine(d, datetime.time(h), tzinfo=NKST)
    target = st.session_state.get("national_requested")
  elif mode == "과거 관측 시연":
    items = demos()
    if items:
      i = st.selectbox("실제 과거 관측 예시", list(range(len(items))),
                       format_func=lambda i: items[i]["label"], key="nat_demo")
      target = items[i]
  _view_at(mode, target, area_id, rain_layer)
  st.caption(
      "실시간·날짜별 전국 강수는 서버에 기상청 API허브 키(AWS 시간통계 활용 승인)가 있어야 표시됩니다."
      " 키가 없어도 기본 공간 순위와 '과거 관측 시연'은 볼 수 있습니다."
  )


# ==========================================
# 4. 연구 결과 요약 (A3 연구 보고서 반영)
# ==========================================
RF_COLOR, LOW_COLOR = "#2a78d6", "#eb6834"


def capture_chart(df, title, height=300):
  fig = px.bar(
      df, x="도시", y="포착률(%)", color="방법", barmode="group", title=title,
      text=df["포착률(%)"].map(lambda v: f"{v:.1f}%"),
      color_discrete_map={"고도 기준(저지대 우선)": LOW_COLOR, "RF 모델": RF_COLOR},
      category_orders={"방법": ["고도 기준(저지대 우선)", "RF 모델"]},
      hover_data={"포착": True, "포착률(%)": ":.1f", "방법": False, "도시": False},
  )
  fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
  fig.update_layout(
      height=height, bargap=0.35, bargroupgap=0.08,
      margin=dict(l=10, r=10, t=50, b=10),
      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
  )
  fig.update_yaxes(range=[0, 100], gridcolor="rgba(128,128,128,.2)", title="상위 5% 안의 침수 기록 비율 (%)")
  fig.update_xaxes(title=None)
  return fig


@st.cache_data
def evidence_tables():
  legacy = pd.read_csv(EVIDENCE / "legacy_equal_area_top5.csv", encoding="utf-8-sig")
  legacy = legacy[legacy.method.isin(["고도 하위", "RF 지형·거리·피복 11개"])].copy()
  legacy["방법"] = legacy.method.map({"고도 하위": "고도 기준(저지대 우선)", "RF 지형·거리·피복 11개": "RF 모델"})
  legacy = legacy.rename(columns={"city": "도시", "capture_percent": "포착률(%)"})
  legacy["포착"] = legacy.captured.astype(str) + " / " + legacy.positive_cells.astype(str)

  nat = pd.read_csv(EVIDENCE / "national_leave_city.csv", encoding="utf-8-sig")
  nat["방법"] = nat.method.map({"elevation_only": "고도 기준(저지대 우선)", "RF_4": "RF 모델"})
  nat = nat.rename(columns={"city": "도시"})
  nat["포착률(%)"] = nat.capture_fraction_top_5 * 100
  nat["포착"] = nat.captured_top_5.astype(str) + " / " + nat.test_positive.astype(str)
  return legacy, nat


def section_results():
  st.markdown("---")
  st.subheader("📄 해안도시 침수 취약도 모델 연구 결과")
  st.caption("2026-11-16 연구 보고서(A3 포스터용) 요약 · 인천(강화군 제외)·강릉·거제")

  st.markdown(
      """
<div style="border-left:4px solid #2a78d6;padding:.6rem 1rem;margin:.4rem 0 1.2rem">
<b>목표</b> · 인천·강릉·거제에서 학습한 침수 취약도 모델이 다른 해안도시에도 적용되는지, 같은 면적을 확인할 때
단순히 저지대부터 살피는 방법보다 침수 기록을 더 많이 찾는지 검증하고, 이를 전국 지도로 조회할 수 있게 한다.<br><br>
<b>결론</b> · 한 도시를 빼고 학습한 1km 모델은 인천에서만 저지대 기준보다 침수 기록을 더 잘 찾았고(상위 5% 포착 0.0→15.2%)
강릉은 같았으며(41.7%) 거제는 낮아(18.2→15.2%), 해안도시 사이에서도 모델 성능이 그대로 옮겨지지 않으므로
전국 지도는 도시별 검증을 전제로 한 참고 도구로 써야 한다.
</div>
""",
      unsafe_allow_html=True,
  )

  st.markdown("#### " + "1. 같은 면적에서 침수 기록을 얼마나 찾았나")
  st.write(
      "모델이 높은 점수를 준 곳부터 분석 격자의 약 5%를 고르고, 그 안에 든 침수 기록의 비율을 계산했습니다."
      " 같은 수의 격자를 고도가 낮은 순서로 고른 결과와 비교했습니다."
  )
  try:
    legacy, nat = evidence_tables()
  except Exception as e:
    st.error(f"검증표(evidence 폴더)를 읽지 못했습니다: {e}")
    return
  c1, c2 = st.columns(2)
  c1.plotly_chart(capture_chart(legacy, "기존 100m · 같은 도시 안 공간 분할 검증"), width="stretch")
  c2.plotly_chart(capture_chart(nat, "전국 1km · 해당 도시를 학습에서 뺀 검증"), width="stretch")
  st.caption(
      "두 그래프는 격자 해상도와 검증 방식이 달라 수치를 서로 직접 비교하지 않습니다."
      " 왼쪽 RF는 지형·거리·토지피복 11변수, 오른쪽 RF는 표면고도·고도차·주변 대비 고도·건축면적 4변수입니다."
  )

  table = nat.pivot_table(index="도시", columns="method", values=["captured_top_5", "ap"], aggfunc="first")
  rows = []
  for city in ["인천", "강릉", "거제"]:
    r = nat[nat["도시"] == city].iloc[0]
    rows.append({
        "평가 도시": city,
        "기록 격자": int(r.test_positive),
        "선택 격자(상위 5%)": int(r.n_top_5),
        "고도 기준 포착": f"{int(table.loc[city, ('captured_top_5', 'elevation_only')])} / {int(r.test_positive)}",
        "RF 포착": f"{int(table.loc[city, ('captured_top_5', 'RF_4')])} / {int(r.test_positive)}",
        "AP 고도 → RF": f"{table.loc[city, ('ap', 'elevation_only')]:.3f} → {table.loc[city, ('ap', 'RF_4')]:.3f}",
    })
  st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
  st.markdown(
      "- 기존 100m 결과의 상위 5%에서 RF는 인천 279개 중 131개, 강릉 167개 중 133개, 거제 355개 중 164개를 포착했습니다."
      " 저지대 기준은 각각 1개, 127개, 222개였습니다.\n"
      "- 1km 도시 제외 평가에서 RF는 인천에서만 저지대 기준보다 포착률과 AP가 높았습니다."
      " 기록 건수가 적어 다른 도시의 성능을 이 결과로 단정할 수 없습니다.\n"
      "- 기존 100m 기상 모델은 별도 호우 3건에서 AP 0.155로, 지형·피복만 쓴 RF(0.184)보다 낮았습니다."
  )

  st.markdown("#### " + "2. 전국 1km 지도와 지수 산출 방법")
  m1, m2, m3 = st.columns(3)
  m1.metric("전국 1km 분석 격자", "105,681개")
  m2.metric("조회 가능 지역", "16개 시도 · 256개 시군구")
  m3.metric("학습 격자 (3도시)", "2,048개 · 기록 78개")
  st.markdown(
      "**기본 공간 순위 S (0~100)** · Copernicus GLO-30 표면고도(평균, 격자 내 고도차, 주변 5×5 격자 최저 대비 고도)와"
      " GHSL 2020 건축면적 비율로 랜덤포레스트를 학습해 전국 격자의 백분위 순위로 표시합니다.\n\n"
      "**강수 반영 참고지수** · 격자에서 30km 이내 가장 가까운 기상청 관측소의 최근 3시간(R3)·12시간(R12) 누적강수를 씁니다."
  )
  st.latex(r"I = S \times \min\left(1,\ \max\left(\frac{R_3}{60},\ \frac{R_{12}}{110}\right)\right)")
  st.caption(
      "60mm·110mm는 기상청 호우주의보 기준에서 가져온 크기 기준값이며, 이 식과 30km 반경은 침수 기록으로 보정한 값이 아닙니다."
      " 예: 인천 지점의 기본 순위 93.6은 2022-06-15 10시(12시간 17.5mm)에 지수 14.9, 2022-08-08 23시(207.3mm)에 93.6이었습니다."
  )

  st.markdown("#### " + "3. 한계와 향후 과제")
  st.markdown(
      "- 침수 기록은 세 도시의 2022년까지 자료이며, 새 도시와 겨울 침수 성능은 평가하지 않았습니다.\n"
      "- 배경 격자는 실제 미침수가 확인된 곳이 아니며, 점수는 침수 확률·침수심·경보가 아닙니다.\n"
      "- 조위, 하천 수위, 배수시설, 토양 포화, 융설은 계산하지 않습니다. 전국 격자 중 957개는 학습 범위 밖 값을 가집니다.\n"
      "- 다음 단계: 새 도시에서 침수 위치·시작 시각이 있는 여러 사건을 확보해 ① 고도 기준 ② 기본 공간 모델"
      " ③ 강수 반영 지수가 같은 면적에서 기록을 얼마나 포착하는지 비교합니다."
  )

  with st.expander("보고서 원본 그림 보기"):
    for name, cap in [
        ("fig_top5_comparison.png", "그림 1. 같은 면적에서의 침수 기록 포착률"),
        ("fig_national_replay.png", "그림 2. 인천 ASOS 한 지점의 실제 과거 관측 재생 (점선: 관측소 30km 반경)"),
    ]:
      p = EVIDENCE / name
      if p.exists():
        try:
          st.image(str(p), caption=cap, width="stretch")
        except Exception:
          st.image(str(p), caption=cap, use_container_width=True)

  with st.expander("자료와 출처"):
    st.markdown(
        "1. 프로젝트 9B·10 검증표 및 해안도시 침수 취약도 연구 결과 보고서 (2026-11-11)\n"
        "2. [SGIS 기반 vuski/admdongkor 경계 ver20261001](https://github.com/vuski/admdongkor/tree/master/ver20261001) · CC BY 4.0\n"
        "3. [Copernicus DEM GLO-30](https://registry.opendata.aws/copernicus-dem/) · Produced using Copernicus WorldDEM-30"
        " © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA\n"
        "4. [Pesaresi & Politis (2023), EC JRC GHS-BUILT-S R2023A](https://doi.org/10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA) · CC BY 4.0\n"
        "5. [기상청 API허브 AWS 시간통계 및 지점정보](https://apihub.kma.go.kr/apiList.do?seqApi=2&seqApiSub=239)\n"
        "6. [기상청 날씨누리 기상특보 발표기준](https://www.weather.go.kr/w/forecast/guide/standard.do)\n"
        "7. [행정안전부 침수흔적도](https://safetydata.go.kr/disaster-data/view?dataSn=108)"
    )


page_three_cities()
section_national()
section_results()
