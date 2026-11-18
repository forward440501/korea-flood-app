"""A single Streamlit component for the nationwide layer."""
from datetime import date, datetime, time, timedelta
import json
import os
from pathlib import Path

import folium
import streamlit as st
from streamlit_folium import st_folium

from national_flood import (
    COP_ATTR, GHSL_ATTR, GRID_DIR, TO_LL,
    area_bounds_ll, cell_at, load_grid, make_view, prepare_pixels, render_pixels,
)
from national_weather import (
    ARCHIVE_SOURCE, KST, KmaAws, Rain, Station, WeatherUnavailable, live_target,
)

ARCHIVE_PATH=GRID_DIR/'asos_replay.json'
NATIONAL_BOUNDS=[[32.95,124.5],[38.70,132.05]]
EXPLANATION=('전국 확장 실험입니다. 3개 도시의 침수 기록으로 학습한 공간 순위와 인근 관측소 강수를 표시합니다. '
             '새 지역·새 계절에서의 침수 예측 성능은 확인되지 않았습니다. 점수는 침수 확률이나 경보가 아닙니다.')
BOUNDARY_NOTE='행정 경계는 자료 기준일의 시군구 단위이며, 클릭 판정에는 200m 토지 마스크를 사용합니다.'


@st.cache_resource
def assets():
    grid=load_grid()
    return grid,prepare_pixels(grid)


@st.cache_resource
def provider(key):
    return KmaAws(key)


def secret_key():
    key=os.environ.get('KMA_APIHUB_KEY','').strip()
    if not key:
        try:
            key=str(st.secrets.get('KMA_APIHUB_KEY','')).strip()
        except (FileNotFoundError,KeyError,ValueError):
            pass
    return key


@st.cache_data
def demos():
    if not ARCHIVE_PATH.is_file():
        return []
    data=json.loads(ARCHIVE_PATH.read_text(encoding='utf-8'))
    return data['examples']


def archive_rain(x):
    t=datetime.fromisoformat(x['end'])
    station=Station(int(x['station']),float(x['lon']),float(x['lat']))
    return Rain(t,ARCHIVE_SOURCE,((station,float(x['rain3']),float(x['rain12'])),))


def _legend(rain_layer):
    st.markdown(
        """
        <div style="font-size:.82rem;line-height:1.45">
          <div style="margin-bottom:6px"><span aria-hidden="true" style="display:inline-block;width:76px;height:9px;
                 background:linear-gradient(90deg,#eceff5,#698ea9,#1c3e66)"></span>
                 기본 공간 순위 0–100</div>
          <div><span aria-hidden="true" style="display:inline-block;width:76px;height:9px;
                 background:linear-gradient(90deg,#f0e7dc,#dc913f,#98242e)"></span>
                 강수 반영 참고지수 0–100</div>
        </div>
        """ if rain_layer else
        """
        <div style="font-size:.82rem"><span aria-hidden="true" style="display:inline-block;width:76px;height:9px;
             background:linear-gradient(90deg,#eceff5,#698ea9,#1c3e66)"></span>
             기본 공간 순위 0–100</div>
        """,unsafe_allow_html=True)


def _map_for(area):
    if area:
        lon,lat=TO_LL.transform(*area['center_xy'])
        m=folium.Map(location=[lat,lon],zoom_start=9,tiles='OpenStreetMap',control_scale=True)
        m.fit_bounds(area_bounds_ll(area),max_zoom=11)
    else:
        m=folium.Map(location=[36.25,127.8],zoom_start=7,tiles='OpenStreetMap',control_scale=True)
        m.fit_bounds(NATIONAL_BOUNDS)
    folium.LatLngPopup().add_to(m)
    return m


def _details(rec):
    if rec is None:
        st.info('이 위치는 1km 분석 격자 밖이거나 200m 경계 마스크에서 확인되지 않습니다. 바다에는 점수를 표시하지 않습니다.')
        return
    st.subheader(rec['area_name'])
    x,y=st.columns(2)
    x.metric('기본 공간 순위',f"{rec['base_percentile']:.1f} / 100")
    y.metric('강수 반영 참고지수', f"{rec['rain_overlay']:.1f} / 100" if rec['rain_overlay'] is not None else '미반영')
    if rec['reason']:
        st.caption(rec['reason'])
    if rec['state']=='rain_applied':
        obs=datetime.fromisoformat(rec['obs_time_kst']).strftime('%Y-%m-%d %H:%M')
        st.write(f"최근 3시간 **{rec['rain3_mm']:.1f} mm** · 최근 12시간 **{rec['rain12_mm']:.1f} mm**")
        st.caption(f"{obs} KST 기준 · {rec['source']} · 지점 {rec['station_id']} · 격자 중심에서 {rec['station_distance_km']:.1f} km")
        if rec['rain3_mm']==0 and rec['rain12_mm']==0:
            st.caption('관측 강수가 0 mm입니다. 이 값은 침수 가능성이 0이라는 뜻이 아닙니다.')
    if rec['outside_marginal_training_range']:
        st.warning('이 격자는 공간 변수 중 하나 이상이 학습에 사용한 세 도시의 값 범위 밖입니다.')
    with st.expander('선택한 격자의 공간 입력'):
        f=rec['inputs']
        st.write(f"평균 표면고도 {f['dem_mean_m']:.1f} m · 격자 내 고도차 {f['dem_relief_m']:.1f} m")
        st.write(f"인근 최저 격자 대비 고도 {f['dem_above_local_low_m']:.1f} m · 건축면적 비율 {f['built_pct']:.1f}%")
        st.caption(f"전국 격자 행·열 {rec['grid_rc'][0]}, {rec['grid_rc'][1]} · 1km · 200m 유효 지형 표본 {rec['support_200m_cells']}/25")


@st.fragment(run_every='5m')
def _view_at(mode,target,area_id,rain_layer):
    grid,pixels=assets()
    now=datetime.now(KST)
    rain=None
    note=None
    if mode=='과거 관측 시연':
        if target is not None:
            rain=archive_rain(target)
            st.info('과거 ASOS 한 지점의 실제 관측을 사용하는 시연입니다. 붉은 계열은 이 지점 근처에만 표시됩니다.')
    elif mode=='실시간 관측' or target:
        obs=live_target(now) if mode=='실시간 관측' else target
        key=secret_key()
        if not key:
            note='서버에 기상청 API허브 키가 연결되지 않았습니다.'
        elif obs>now:
            note='미래 시각은 관측으로 조회할 수 없습니다.'
        else:
            try:
                rain=provider(key).snapshot(obs,live_now=now if mode=='실시간 관측' else None)
            except WeatherUnavailable as exc:
                note=str(exc)
    else:
        note='조회 날짜와 시각을 고르고 조회를 눌러 주세요.'
    if mode=='날짜·시간 조회' and target:
        st.caption(f"요청한 관측 종료 시각 {target:%Y-%m-%d %H:%M} KST")
    v=make_view(grid,rain,reason=note)
    if rain is not None and rain.readings:
        timestamp=rain.end.strftime('%Y-%m-%d %H:%M')
        st.caption(f'{rain.source} · 관측 종료 {timestamp} KST · '
                   f'사용 관측소 {len(rain.readings):,}개 · '
                   f'강수 반영 격자 {int(v.observed.sum()):,}/{grid.manifest["cells"]:,}개')
        if v.reason: st.caption(v.reason)
    else:
        st.info(f"기본 공간 순위를 표시 중입니다. {v.reason or '관측을 선택해 주세요.'}")
    area=grid.areas.get(area_id)
    if area:
        st.caption(f'{area["province"]} {area["name"]} · 해당 경계와 겹치는 분석 격자 {area["n_grids"]:,}개')
    m,info=st.columns([2.25,1],gap='large')
    with m:
        rgba,bounds=render_pixels(v,pixels,rain_layer=rain_layer)
        layer=folium.FeatureGroup(name='전국 침수 참고지도',overlay=True,control=False)
        folium.raster_layers.ImageOverlay(rgba,bounds=bounds,origin='upper',opacity=.93,
            name='전국 침수 참고지도',interactive=False,mercator_project=False,show=True).add_to(layer)
        map_event=st_folium(_map_for(area),key=f'nat_map_{area_id}',feature_group_to_add=layer,
            height=540,use_container_width=True,returned_objects=['last_clicked'],pixelated=True)
        _legend(rain_layer)
    with info:
        point_key=f'nat_saved_point_{area_id}'
        pt=(map_event or {}).get('last_clicked')
        if pt:
            st.session_state[point_key]=pt
        else:
            pt=st.session_state.get(point_key)
        if pt:
            rec=cell_at(v,float(pt['lat']),float(pt['lng']))
            _details(rec)
        else:
            st.subheader('지점 자세히 보기')
            st.write('지도에서 관심 위치를 클릭하세요. 기본 공간 순위, 적용한 관측소 및 강수량을 확인할 수 있습니다.')
    st.caption(f"행정 경계: {grid.manifest['boundary_source']} · {grid.manifest['boundary_date']} · {BOUNDARY_NOTE}")
    st.caption(f'공간자료: {GHSL_ATTR} | {COP_ATTR}')


def render_national_flood():
    g,_=assets()
    st.title('대한민국 침수 참고지도')
    st.caption(EXPLANATION)
    areas=g.areas
    c1,c2,c3=st.columns([1.2,1.5,1])
    province=c1.selectbox('시도', ['전국']+sorted({a['province'] for a in areas.values()}),
                          key='national_province')
    selectable=[a for a in areas.values() if province=='전국' or a['province']==province]
    options=[0]+[a['id'] for a in sorted(selectable,key=lambda a:(a['province'],a['name']))]
    area_id=c2.selectbox('시군구 찾기', options, format_func=lambda a:
        f"{areas[a]['province']} {areas[a]['name']}" if a else '전국 지도',key='national_area')
    mode=c3.selectbox('조회 모드',['실시간 관측','날짜·시간 조회','과거 관측 시연'],key='national_mode')
    rain_layer=st.toggle('지도에 강수 반영 지수 표시',value=True,key='national_rain_layer')
    if province!='전국' and area_id==0:
        st.caption('시군구를 선택하면 해당 지역으로 이동합니다.')
    target=None
    if mode=='날짜·시간 조회':
        latest=datetime.now(KST).date()
        with st.form('national_date_form'):
            a,b,c=st.columns([1.5,1,1.2],vertical_alignment='bottom')
            d=a.date_input('관측 날짜',value=latest,min_value=date(1997,1,1),max_value=latest,
                format='YYYY-MM-DD',key='national_archive_date')
            h=b.selectbox('정시 (KST)',tuple(range(24)),format_func=lambda h:f'{h:02d}:00',
                index=live_target(datetime.now(KST)).hour,key='national_archive_hour')
            if c.form_submit_button('관측 조회'):
                st.session_state['national_requested']=datetime.combine(d,time(h),tzinfo=KST)
        target=st.session_state.get('national_requested')
        st.caption('시각 제약을 두지 않습니다. 실제 과거 조회 범위는 관측소 운영 기간과 API 제공 범위에 따릅니다.')
    elif mode=='과거 관측 시연':
        items=demos()
        if items:
            i=st.selectbox('실제 과거 관측 예시',list(range(len(items))),format_func=lambda i:items[i]['label'],key='nat_demo')
            target=items[i]
    _view_at(mode,target,area_id,rain_layer)
