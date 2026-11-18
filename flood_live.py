"""Research score adapter for the three existing 100m city grids."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer

from table_io import read_table
from weather import KST, STATIONS, SEASON, WeatherIssue, features, normalize, time_kst

GRID_COUNTS = {"인천": 62182, "강릉": 104048, "거제": 40979}
BASE_COLS = ["cell_id", "city", "x_m", "y_m"]
PALETTE = np.array([[44, 80, 158], [48, 172, 164], [250, 165, 64]], dtype=float)
BASELINE_PALETTE = np.array([[225, 234, 241], [111, 144, 168], [38, 67, 92]], dtype=float)


@dataclass(frozen=True)
class YearRoundArtifacts:
    weather_model: tuple
    base_by_city: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class ScoreSnapshot:
    issue: pd.Timestamp
    kind: str
    scores: pd.DataFrame
    reason: str | None
    last_obs: pd.Timestamp | None
    records: pd.DataFrame
    inputs: pd.Series | None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_artifacts(path):
    root = Path(path)
    meta = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    artifact = root / meta["model_path"]
    grid = root / "grid_9b.parquet"
    if digest(artifact) != meta["model_sha256"] or digest(grid) != meta["grid_sha256"]:
        raise ValueError("모델 또는 격자 파일이 검증된 파일과 다릅니다.")
    if meta["status"] != "research_only" or meta["station_map"] != STATIONS:
        raise ValueError("모델 범위 및 표시 정책이 이 플랫폼 버전과 일치하지 않습니다.")
    if meta["algorithm"] == "RF":
        import sklearn
        if meta["scikit_learn_version"] != sklearn.__version__:
            raise ValueError("모델 학습에 사용한 scikit-learn 버전으로 설치해 주세요.")
        # Only load the trusted local model shipped with this handoff.
        est = joblib.load(artifact)
    elif meta["algorithm"] == "XGBoost":
        from xgboost import XGBClassifier
        est = XGBClassifier()
        est.load_model(artifact)
    else:
        raise ValueError("알 수 없는 모델 형식입니다.")
    g = read_table(grid)
    if g.groupby("city").size().to_dict() != GRID_COUNTS or not g.cell_id.is_unique:
        raise ValueError("9B 분석 범위가 다릅니다.")
    return meta, est, g


def load_yearround_artifacts(path):
    root = Path(path)
    bundle = load_artifacts(root)
    meta, _, grid = bundle
    if "year_round" not in meta:
        raise ValueError("연중 표시 버전의 artifacts 폴더를 함께 교체해 주세요.")
    config = meta["year_round"]
    base_path = root / config["base_path"]
    if digest(base_path) != config["base_sha256"]:
        raise ValueError("기본 점수 파일이 검증된 파일과 다릅니다.")
    base = read_table(base_path)
    if (not base.cell_id.equals(grid.cell_id) or
            not np.isfinite(base.score).all() or not base.score.between(0, 100).all()):
        raise ValueError("기본 점수가 기존 격자와 일치하지 않거나 값의 범위를 벗어납니다.")
    g = grid[BASE_COLS].assign(score=base.score.to_numpy())
    return YearRoundArtifacts(bundle, {city: part.copy() for city, part in g.groupby("city", sort=False)})


def load_archive(path, artifacts):
    config = artifacts.weather_model[0]["year_round"]
    p = Path(path) / config["archive_path"]
    if digest(p) != config["archive_sha256"]:
        raise ValueError("과거 ASOS 관측 파일이 검증된 파일과 다릅니다.")
    return read_table(p)


def observation_status(city, records, issue):
    st = STATIONS[city]
    t = time_kst([issue]).iloc[0].floor("h")
    cutoff = t - pd.Timedelta(hours=1)
    if records.empty:
        raise WeatherIssue("해당 지점에서 받은 기상 관측이 없습니다.")
    obs = normalize(records)
    obs = obs.loc[obs.station.eq(st) &
                  obs.time_kst.between(t - pd.Timedelta(hours=30), cutoff)].copy()
    if obs.empty:
        raise WeatherIssue("해당 지점의 기준시각 이전 관측이 없습니다.")
    latest = obs.time_kst.max()
    state = {"station": st, "issue": t, "last_obs": latest, "reason": None, "input": None}
    if latest != cutoff:
        state["reason"] = "기준시각 직전의 정시 관측이 아직 도착하지 않았습니다."
        return state, obs
    if t.month not in SEASON or (t - pd.Timedelta(hours=24)).month not in SEASON:
        state["reason"] = "기상 모델에 사용한 7~10월 범위 밖입니다."
        return state, obs
    row = features(obs)[st].loc[t]
    if row.station_rows_24h != 24:
        state["reason"] = "최근 24시간 중 빠진 정시 관측이 있습니다."
        return state, obs
    if row.rain_value_hours_24h == 0:
        state["reason"] = "최근 24시간에 수치가 있는 강수 관측이 없습니다. 공백을 0 mm로 바꾸지 않습니다."
        return state, obs
    state["input"] = row
    return state, obs


def predict_grid(city, state, bundle):
    if state["input"] is None or state["reason"] is not None:
        raise WeatherIssue(state["reason"] or "기상 입력을 확인해 주세요.")
    if state["station"] != STATIONS[city]:
        raise WeatherIssue("모델 입력 지점이 선택한 도시와 일치하지 않습니다.")
    meta, estimator, grid = bundle
    g = grid.loc[grid.city.eq(city)].copy()
    cols = meta["features"]
    x = g.reindex(columns=cols)
    for name in cols:
        if name not in g:
            x[name] = float(state["input"][name])
    pred = estimator.predict_proba(x)[:, 1]
    if not np.isfinite(pred).all() or not np.all((pred >= 0) & (pred <= 1)):
        raise WeatherIssue("모델이 유효한 점수를 반환하지 않았습니다.")
    return g[BASE_COLS].assign(score=(100 * pred))


def score_snapshot(city, issue, artifacts, records=None, *, observation_error=None):
    """Return one score per analyzed grid, even when no weather is available.

    'baseline' and 'weather' are separate models; never treat a switch between
    them as a measured change in flood risk.
    """
    issue = time_kst([issue]).iloc[0].floor("h")
    baseline = artifacts.base_by_city[city]
    if observation_error or records is None:
        return ScoreSnapshot(issue, "baseline", baseline,
                             observation_error or "이 시각에 연결된 기상 관측이 없습니다.",
                             None, pd.DataFrame(), None)
    try:
        state, cleaned = observation_status(city, records, issue)
    except WeatherIssue as exc:
        return ScoreSnapshot(issue, "baseline", baseline, str(exc), None, pd.DataFrame(), None)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return ScoreSnapshot(issue, "baseline", baseline, "기상 관측 자료의 형식이 맞지 않습니다.",
                             None, pd.DataFrame(), None)
    if state["input"] is None:
        return ScoreSnapshot(issue, "baseline", baseline, state["reason"],
                             state["last_obs"], cleaned, None)
    scores = predict_grid(city, state, artifacts.weather_model)
    return ScoreSnapshot(issue, "weather", scores, None, state["last_obs"], cleaned, state["input"])


def locate(g, lat, lon):
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    x, y = Transformer.from_crs(4326, 5179, always_xy=True).transform(lon, lat)
    inside = g.loc[(g.x_m - 50 <= x) & (x < g.x_m + 50) &
                   (g.y_m - 50 <= y) & (y < g.y_m + 50)]
    return inside.iloc[0] if len(inside) == 1 else None


def map_raster(g, kind="weather"):
    """Project the 5179 grid to a display image on Leaflet's Web Mercator map."""
    xx = g.x_m.to_numpy()
    yy = g.y_m.to_numpy()
    lx, ly = xx.min() - 50, yy.min() - 50
    wx, wy = xx.max() + 50, yy.max() + 50
    grid_x = ((xx - lx) // 100).astype(int)
    grid_y = ((yy - ly) // 100).astype(int)
    v = np.full((grid_y.max() + 1, grid_x.max() + 1), np.nan)
    v[grid_y, grid_x] = g.score.to_numpy() / 100
    merc = Transformer.from_crs(5179, 3857, always_xy=True)
    inverse = Transformer.from_crs(3857, 5179, always_xy=True)
    wgs = Transformer.from_crs(3857, 4326, always_xy=True)
    lo_x, lo_y, hi_x, hi_y = merc.transform_bounds(lx, ly, wx, wy, densify_pts=21)
    nx, ny = max(2, int(np.ceil((hi_x - lo_x) / 90))), max(2, int(np.ceil((hi_y - lo_y) / 90)))
    # Centre each pixel; target rows run from north to south.
    dx, dy = (hi_x - lo_x) / nx, (hi_y - lo_y) / ny
    x, y = np.meshgrid(lo_x + (np.arange(nx) + .5) * dx, hi_y - (np.arange(ny) + .5) * dy)
    sx, sy = inverse.transform(x, y)
    gx, gy = np.floor((sx-lx)/100).astype(int), np.floor((sy-ly)/100).astype(int)
    ok = (gx >= 0) & (gy >= 0) & (gx < v.shape[1]) & (gy < v.shape[0])
    val = np.full_like(x, np.nan)
    val[ok] = v[gy[ok], gx[ok]]
    rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
    valid = np.isfinite(val)
    palette = BASELINE_PALETTE if kind == "baseline" else PALETTE
    for i in range(3):
        rgba[..., i][valid] = np.interp(val[valid], (0, .5, 1), palette[:, i]).astype(np.uint8)
    rgba[..., 3][valid] = 177
    lon0, lat0 = wgs.transform(lo_x, lo_y)
    lon1, lat1 = wgs.transform(hi_x, hi_y)
    return rgba, [[lat0, lon0], [lat1, lon1]]
