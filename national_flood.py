"""Countrywide 1-km screening layer, separate from the original three-city model."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from math import isfinite
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

from national_weather import Rain, Station

_HERE=Path(__file__).resolve().parent
# 폴더 구조가 있으면 artifacts_national/manifest.json, 한 곳에 모은 배포판이면 national_manifest.json
GRID_DIR=_HERE/'artifacts_national' if (_HERE/'artifacts_national').is_dir() else _HERE
MANIFEST_NAME='manifest.json' if GRID_DIR.name=='artifacts_national' else 'national_manifest.json'
MATCH_RADIUS_M=30_000.0
RAIN3_SCALE_MM=60.0
RAIN12_SCALE_MM=110.0
TO_XY=Transformer.from_crs(4326,5179,always_xy=True)
TO_LL=Transformer.from_crs(5179,4326,always_xy=True)
COP_ATTR=('Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH '
          '2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved')
GHSL_ATTR='EC JRC GHS-BUILT-S R2023A (Pesaresi & Politis, 2023), CC BY 4.0, 가공'
BASE_COLORS=np.array([[236,241,248],[105,142,169],[28,62,102]],dtype=float)
LIVE_COLORS=np.array([[240,231,220],[220,145,63],[152,36,46]],dtype=float)


@dataclass(frozen=True)
class Grid:
    manifest: dict
    score: np.ndarray
    features: np.ndarray
    support_n: np.ndarray
    fine_admin: np.ndarray
    fine_valid: np.ndarray
    extrap: np.ndarray
    valid_rows: np.ndarray
    valid_cols: np.ndarray
    centers: np.ndarray
    # Indexed by stable integer raster ID (zero means outside).
    areas: dict[int,dict]


@dataclass(frozen=True)
class View:
    grid: Grid
    rain: Rain | None
    observed: np.ndarray
    gauge_index: np.ndarray
    distance_km: np.ndarray
    combined: np.ndarray
    reason: str | None
    radius_km: float


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1<<20):
            h.update(b)
    return h.hexdigest()


def load_grid(path=GRID_DIR):
    root=Path(path)
    m=json.loads((root/MANIFEST_NAME).read_text(encoding='utf-8'))
    if m['resolution_m']!=1000 or m['land_check_m']!=200:
        raise ValueError('전국 격자의 해상도가 예상과 다릅니다.')
    p=root/'national_grid.npz'
    if file_hash(p)!=m['grid_sha256']:
        raise ValueError('전국 연구 격자 파일의 무결성을 확인하지 못했습니다.')
    with np.load(p,allow_pickle=False) as ds:
        attrs={k:ds[k] for k in ds.files}
    score=attrs['score']
    fine=attrs['fine_admin']
    valid=attrs['fine_valid']
    areas={a['id']:a for a in m['areas']}
    if (score.shape!=tuple(m['shape']) or int(np.isfinite(score).sum())!=m['cells'] or
        fine.shape!=tuple(n*5 for n in m['shape']) or fine.shape!=valid.shape or
        attrs['features'].shape!=(len(m['feature_names']),*score.shape) or
        attrs['extrap'].shape!=score.shape or attrs['support_n'].shape!=score.shape or
        len(areas)!=m['regions'] or fine.max()>len(areas) or
        not np.all((score[np.isfinite(score)]>=0)&(score[np.isfinite(score)]<=100))):
        raise ValueError('전국 격자와 분석 범위 레코드가 일치하지 않습니다.')
    rows,cols=np.where(np.isfinite(score))
    x0,_,_,y0=m['bounds_5179']
    centers=np.column_stack([x0+(cols+.5)*1000,y0-(rows+.5)*1000])
    for a in [*attrs.values(),rows,cols,centers]:
        a.setflags(write=False)
    return Grid(m,**attrs,valid_rows=rows,valid_cols=cols,centers=centers,areas=areas)


def rain_factor(rain3,rain12):
    return np.minimum(1.,np.maximum(np.asarray(rain3)/RAIN3_SCALE_MM,np.asarray(rain12)/RAIN12_SCALE_MM))


def make_view(grid,rain=None,*,reason=None,radius_m=MATCH_RADIUS_M):
    if not isfinite(radius_m) or not 0<radius_m<=100_000:
        raise ValueError('관측소 연결 거리는 0~100km 범위에서 지정해 주세요.')
    score=grid.score
    observed=np.zeros(score.shape,dtype=bool)
    gi=np.full(score.shape,-1,dtype=np.int32)
    km=np.full(score.shape,np.nan,dtype=np.float32)
    combo=np.full(score.shape,np.nan,dtype=np.float32)
    if rain is not None and rain.readings:
        rr=[]; xs=[]; ys=[]; factors=[]; used_ids=set()
        for i,(t,p3,p12) in enumerate(rain.readings):
            if not (isinstance(t,Station) and t.id not in used_ids and
                    isfinite(t.lon) and isfinite(t.lat) and 124.<=t.lon<=132.1 and 32.5<=t.lat<=39. and
                    isfinite(p3) and isfinite(p12) and 0<=p3<=p12):
                raise ValueError('기상 스냅샷의 관측소 좌표 또는 강수량이 유효하지 않습니다.')
            used_ids.add(t.id)
            x,y=TO_XY.transform(t.lon,t.lat)
            rr.append(i);xs.append(x);ys.append(y);factors.append(float(rain_factor(p3,p12)))
        tree=cKDTree(np.column_stack((xs,ys)))
        distances,nearest=tree.query(grid.centers,k=1,distance_upper_bound=radius_m)
        ok=np.isfinite(distances)
        r,c=grid.valid_rows[ok],grid.valid_cols[ok]
        s=nearest[ok]
        gi[r,c]=np.asarray(rr,dtype='int32')[s]
        km[r,c]=distances[ok]/1000.
        observed[r,c]=True
        combo[r,c]=score[r,c]*np.asarray(factors)[s]
    for a in (observed,gi,km,combo):a.setflags(write=False)
    return View(grid,rain,observed,gi,km,combo,
                reason or (rain.reason if rain else None),radius_m/1000)


def area_bounds_ll(area):
    lo_x,lo_y,hi_x,hi_y=area['bounds']
    lon1,lat1,lon2,lat2=TO_LL.transform_bounds(lo_x,lo_y,hi_x,hi_y,densify_pts=9)
    return [[lat1,lon1],[lat2,lon2]]


def cell_at(view,lat,lon):
    """Portable return contract for the teammate's map. Returns None outside analysis coverage."""
    if not (isfinite(lat) and isfinite(lon) and 32.5<=lat<=39 and 124<=lon<=132.1):
        return None
    g=view.grid
    xmin,_,_,ymax=g.manifest['bounds_5179']
    x,y=TO_XY.transform(lon,lat)
    i,j=int(np.floor((ymax-y)/200)),int(np.floor((x-xmin)/200))
    if not (0<=i<g.fine_admin.shape[0] and 0<=j<g.fine_admin.shape[1] and g.fine_valid[i,j]):
        return None
    ar=g.areas[int(g.fine_admin[i,j])]
    r,c=i//5,j//5
    p={'area_code':ar['code'],'area_name':f"{ar['province']} {ar['name']}",'grid_rc':[r,c],
       'grid_m':1000,'base_percentile':float(g.score[r,c]),
       'outside_marginal_training_range':bool(g.extrap[r,c]),
       'support_200m_cells':int(g.support_n[r,c]),
       'inputs':{key:float(v) for key,v in zip(g.manifest['feature_names'],g.features[:,r,c])},
       'rain_overlay':None,'source':None,'obs_time_kst':None,'station_id':None,
       'station_distance_km':None,'rain3_mm':None,'rain12_mm':None,
       'state':'base_only','reason':None}
    if view.observed[r,c]:
        obs=view.rain
        assert obs is not None
        station,p3,p12=obs.readings[view.gauge_index[r,c]]
        p.update(rain_overlay=float(view.combined[r,c]),source=obs.source,obs_time_kst=obs.end.isoformat(),
            station_id=station.id,station_distance_km=float(view.distance_km[r,c]),
            rain3_mm=p3,rain12_mm=p12,state='rain_applied',reason=view.reason)
    else:
        p['reason']=('거리 '+str(round(view.radius_km))+' km 안에 연속 12시간 강수가 확인된 관측소가 없습니다.'
                     if view.rain and view.rain.readings else
                     view.reason or '연결된 강수 관측이 없습니다.')
    return p


@dataclass(frozen=True)
class ProjectedPixels:
    rows: np.ndarray
    cols: np.ndarray
    inside: np.ndarray
    bounds: list


def prepare_pixels(g,metres=900):
    """Leaflet overlay image in Web Mercator; EPSG:5179 fine mask controls the shoreline."""
    x0,y0,x1,y1=g.manifest['bounds_5179']
    to_merc=Transformer.from_crs(5179,3857,always_xy=True)
    to_utm=Transformer.from_crs(3857,5179,always_xy=True)
    to_ll=Transformer.from_crs(3857,4326,always_xy=True)
    xmin,ymin,xmax,ymax=to_merc.transform_bounds(x0,y0,x1,y1,densify_pts=21)
    nx=int(np.ceil((xmax-xmin)/metres));ny=int(np.ceil((ymax-ymin)/metres))
    x,y=np.meshgrid(xmin+(np.arange(nx)+.5)*(xmax-xmin)/nx,ymax-(np.arange(ny)+.5)*(ymax-ymin)/ny)
    ux,uy=to_utm.transform(x,y)
    r=np.floor((y1-uy)/200).astype(np.int32)
    c=np.floor((ux-x0)/200).astype(np.int32)
    inside=(r>=0)&(c>=0)&(r<g.fine_valid.shape[0])&(c<g.fine_valid.shape[1])
    r=np.clip(r,0,g.fine_valid.shape[0]-1);c=np.clip(c,0,g.fine_valid.shape[1]-1)
    inside&=g.fine_valid[r,c]
    lon1,lat1=to_ll.transform(xmin,ymin)
    lon2,lat2=to_ll.transform(xmax,ymax)
    return ProjectedPixels(r//5,c//5,inside,[[lat1,lon1],[lat2,lon2]])


def render_pixels(view,pixels,*,rain_layer=True):
    r,c=pixels.rows,pixels.cols
    ok=pixels.inside
    has=ok&view.observed[r,c]&rain_layer
    base=ok&~has
    image=np.zeros((*ok.shape,4),dtype=np.uint8)
    for mask,values,colors in [(base,view.grid.score,BASE_COLORS),(has,view.combined,LIVE_COLORS)]:
        v=values[r[mask],c[mask]]
        for i in (0,1,2):
            image[:,:,i][mask]=np.interp(v,[0.,50.,100.],colors[:,i]).astype(np.uint8)
        image[:,:,3][mask]=195
    return image,pixels.bounds
