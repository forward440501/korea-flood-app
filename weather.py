"""One feature transform for the Drive ASOS archive and the live feed."""
from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

KST = "Asia/Seoul"
STATIONS = {"인천": 112, "강릉": 105, "거제": 294}
SEASON = (7, 8, 9, 10)
ENDPOINT = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
ARCHIVE = {
    "지점": "station", "일시": "time_kst", "강수량(mm)": "rain_mm",
    "기온(°C)": "temp_c", "습도(%)": "rh_pct", "해면기압(hPa)": "slp_hpa",
    "풍속(m/s)": "wind_m_s",
}
RAW = ["rain_mm", "temp_c", "rh_pct", "slp_hpa", "wind_m_s"]
RAIN = [
    "rain_mm_1h", "rain_known_sum_3h", "rain_known_sum_6h",
    "rain_known_sum_24h", "rain_known_max_6h",
    "rain_value_hours_3h", "rain_value_hours_6h", "rain_value_hours_24h",
]
OTHER = ["temp_c", "rh_pct", "slp_hpa", "wind_m_s", "slp_change_3h", "wind_max_6h"]


class WeatherIssue(ValueError):
    pass


def csv(path, **kwargs):
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            pass
    raise WeatherIssue(f"CSV 인코딩을 읽을 수 없습니다: {Path(path).name}")


def time_kst(values):
    values = pd.Series(values).reset_index(drop=True)
    if values.empty:
        return pd.Series([], dtype=f"datetime64[ns, {KST}]")
    if not pd.api.types.is_datetime64_any_dtype(values):
        compact = values.astype("string").str.fullmatch(r"\d{12}").fillna(False)
        if compact.all():
            values = pd.to_datetime(values, format="%Y%m%d%H%M", errors="raise")
        else:
            values = pd.to_datetime(values, format="mixed", errors="raise")
    if not pd.api.types.is_datetime64_any_dtype(values):
        raise WeatherIssue("관측시각의 표준시 표기가 일관되지 않습니다.")
    return values.dt.tz_localize(KST) if values.dt.tz is None else values.dt.tz_convert(KST)


def normalize(raw):
    """Only exact hourly readings; keep blanks/negative rain codes missing."""
    req = {"station", "time_kst", *RAW}
    if not req.issubset(raw.columns):
        raise WeatherIssue(f"기상 필수 열 누락: {', '.join(sorted(req - set(raw.columns)))}")
    d = raw[["station", "time_kst", *RAW]].copy().reset_index(drop=True)
    d["station"] = pd.to_numeric(d.station, errors="raise").astype("Int64")
    d = d.loc[d.station.isin(STATIONS.values())].reset_index(drop=True)
    d["time_kst"] = time_kst(d.time_kst)
    if d.empty or d.time_kst.isna().any() or not d.time_kst.eq(d.time_kst.dt.floor("h")).all():
        raise WeatherIssue("선택한 지점의 정시 관측이 없거나 시각이 유효하지 않습니다.")
    bounds = {"rain_mm": (0, 500), "temp_c": (-60, 60), "rh_pct": (0, 100),
              "slp_hpa": (870, 1100), "wind_m_s": (0, 100)}
    for c, (lo, hi) in bounds.items():
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
        d[c] = d[c].where(d[c].between(lo, hi))
    d = d.drop_duplicates()
    duplicate = d.duplicated(["station", "time_kst"], keep=False)
    if duplicate.any():
        # Resolve overlaps only if no actual observed value conflicts.
        mixed = d.loc[duplicate].groupby(["station", "time_kst"])[RAW]
        if mixed.nunique(dropna=True).gt(1).any().any():
            raise WeatherIssue("같은 ASOS 지점·시각에 서로 다른 관측값이 있습니다.")
        d = pd.concat([d.loc[~duplicate], mixed.first().reset_index()], ignore_index=True)
    return d.sort_values(["station", "time_kst"]).reset_index(drop=True)


def read_asos(folder):
    paths = sorted(Path(folder).glob("ASOS_*.csv"))
    if not paths:
        raise WeatherIssue("기상자료 폴더에 ASOS_*.csv가 없습니다.")
    frames = [csv(p, usecols=list(ARCHIVE)).rename(columns=ARCHIVE) for p in paths]
    return normalize(pd.concat(frames, ignore_index=True)), paths, sum(map(len, frames))


def features(raw):
    """At issue hour t, use only records ending at or before t minus one hour."""
    norm = normalize(raw)
    result = {}
    for st, d in norm.groupby("station", sort=True):
        full = pd.date_range(d.time_kst.iloc[0], d.time_kst.iloc[-1] + pd.Timedelta(hours=1), freq="h")
        a = d.set_index("time_kst").reindex(full)
        f = a[RAW].shift(1).copy()
        f = f.rename(columns={"rain_mm": "rain_mm_1h"})
        f["slp_change_3h"] = a.slp_hpa.shift(1) - a.slp_hpa.shift(4)
        f["wind_max_6h"] = a.wind_m_s.shift(1).rolling(6, min_periods=1).max()
        for h in (3, 6, 24):
            v = f.rain_mm_1h.rolling(h, min_periods=1)
            f[f"rain_known_sum_{h}h"] = v.sum()
            f[f"rain_value_hours_{h}h"] = f.rain_mm_1h.notna().rolling(h, min_periods=1).sum()
        f["rain_known_max_6h"] = f.rain_mm_1h.rolling(6, min_periods=1).max()
        f["station_rows_24h"] = a.station.notna().shift(1, fill_value=False).rolling(24, min_periods=1).sum()
        # The whole preceding window must be inside the supported season.
        f["in_season"] = f.index.month.isin(SEASON) & (f.index - pd.Timedelta(hours=24)).month.isin(SEASON)
        f.index.name = "issue_time_kst"
        result[int(st)] = f
    return result


def parse_kma(body):
    # Accept both documented variable labels and header in the actual ASCII response.
    # Never infer RN from a positional column if its header has not been seen.
    header = None
    rows = []
    names = {"STN": "station", "RN": "rain_mm", "TA": "temp_c",
             "HM": "rh_pct", "PS": "slp_hpa", "WS": "wind_m_s"}
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            cols = line.lstrip("#").split()
            if cols and cols[0] in {"YYMMDDHHMI", "YYYYMMDDHHMI", "TM"} and set(names) <= set(cols):
                header = cols
            continue
        cols = line.split()
        if not re.fullmatch(r"\d{12}", cols[0]) or header is None or len(cols) != len(header):
            raise WeatherIssue("기상청 응답에서 공식 시간관측 표를 확인하지 못했습니다. 인증키, 활용신청, 응답 형식을 확인해 주세요.")
        rows.append({"time_kst": cols[0], **{key: cols[header.index(k)] for k, key in names.items()}})
    if not rows:
        raise WeatherIssue("기상청 응답에 시간관측 행이 없습니다.")
    return normalize(pd.DataFrame(rows))


def fetch_kma(key, station, issue_time, hours=30):
    """KMA API Hub hourly ASOS; no key or key-containing URL is logged."""
    if not key:
        raise WeatherIssue("서버 설정에 KMA_APIHUB_KEY가 필요합니다.")
    if station not in STATIONS.values() or not 25 <= hours <= 168:
        raise WeatherIssue("지점 또는 조회 기간을 확인해 주세요.")
    issue = time_kst([issue_time]).iloc[0].floor("h")
    end = issue - pd.Timedelta(hours=1)
    start = issue - pd.Timedelta(hours=hours)
    params = {"tm1": start.strftime("%Y%m%d%H%M"), "tm2": end.strftime("%Y%m%d%H%M"),
              "stn": station, "help": 1, "authKey": key}
    try:
        req = Request(f"{ENDPOINT}?{urlencode(params)}", headers={"User-Agent": "CoastalFloodResearch"})
        with urlopen(req, timeout=15) as response:
            body = response.read(2_000_000).decode("utf-8", "replace")
    except Exception as exc:
        raise WeatherIssue(f"기상청 연결에 실패했습니다 ({type(exc).__name__}).") from None
    frame = parse_kma(body)
    return frame.loc[frame.station.eq(station) & frame.time_kst.between(start, end)].copy()
