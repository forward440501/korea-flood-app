"""Year-round KMA AWS hourly rainfall. Requests and API keys stay on the server."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from threading import Lock
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from zoneinfo import ZoneInfo
import re
import time

KST = ZoneInfo('Asia/Seoul')
HOUR = timedelta(hours=1)
AWS_HOURLY = 'https://apihub.kma.go.kr/api/typ01/url/awsh.php'
KMA_STATIONS = 'https://apihub.kma.go.kr/api/typ01/url/stn_inf.php'
PUBLIC_SOURCE = '기상청 API허브 AWS 시간통계'
ARCHIVE_SOURCE = '기존 연구 ASOS 과거 관측 시연'
MAX_RESPONSE = 2_000_000
READY_MINUTES = 10
MAX_LIVE_MINUTES = 100


class WeatherUnavailable(Exception):
    """Only user-safe messages; never include request URLs or tokens."""


@dataclass(frozen=True)
class Station:
    id: int
    lon: float
    lat: float


@dataclass(frozen=True)
class Rain:
    end: datetime
    source: str
    # Rainfall windows end at 'end'; both windows must be complete at this gauge.
    readings: tuple[tuple[Station,float,float], ...]
    reason: str | None = None


def hourstamp(t):
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError('조회 시각의 시간대를 지정해 주세요.')
    return t.astimezone(KST).replace(minute=0,second=0,microsecond=0)


def live_target(now):
    return hourstamp(now-timedelta(minutes=READY_MINUTES))


def tm(t):
    return t.astimezone(KST).strftime('%Y%m%d%H%M')


def number(s):
    try:
        v=float(s)
    except (ValueError,TypeError):
        return None
    return v if isfinite(v) else None


def tokens(s):
    s=s.strip().replace('−','-').replace('–','-')
    if ',' in s:
        return [x.strip() for x in s.split(',')]
    return s.split()


def checked_text(raw):
    if not raw or len(raw)>MAX_RESPONSE:
        raise WeatherUnavailable('관측 API 응답 크기가 비정상입니다.')
    for enc in ('utf-8-sig','cp949'):
        try:
            s=raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise WeatherUnavailable('관측 API의 문자 형식을 읽을 수 없습니다.')
    lead=s.lstrip()
    if lead.startswith(('<','{','[')):
        raise WeatherUnavailable('기상청이 표 대신 오류 응답을 반환했습니다. 활용 승인과 호출 한도를 확인하세요.')
    return s


def parse_rn(raw,end):
    """Read KMA awsh.php var=RN. RN_HR1 is column 7 in official grouped header.

    Also supports the explicit RN_HR1/RN_HR1_MI machine-style header. No
    data-only layout guessing. Negative/sentinel values and incomplete hours
    are omitted; zero with complete window remains a valid observation.
    """
    result={}
    excluded=set()
    schema=None
    ts=tm(end)
    for line in checked_text(raw).splitlines():
        line=line.strip()
        if not line:
            continue
        t=tokens(line.lstrip('#'))
        u=[x.upper() for x in t]
        if ('RN_HR1' in u) and ('RN_HR1_MI' in u) and ('STN' in u):
            if len(u)>=4 and u[0] in ('YYMMDDHHMI','YYYYMMDDHHMI','TM'):
                schema=(len(u),u.index('STN'),u.index('RN_HR1'),u.index('RN_HR1_MI'))
            continue
        if (len(u)==14 and u[0] in ('YYMMDDHHMI','YYYYMMDDHHMI')
            and u[1:8]==['STN','RE','RE','RN','MI','RN','MI']):
            schema=(14,1,6,7)
            continue
        if line.startswith('#') or not t or t[0]!=ts:
            continue
        if schema is None:
            raise WeatherUnavailable('기상청 AWS 시간강수 열을 확인하지 못했습니다.')
        n,id_i,rn_i,mi_i=schema
        if len(t)!=n or not t[id_i].isdigit():
            continue
        st=int(t[id_i])
        val=number(t[rn_i])
        win=number(t[mi_i])
        if not (1 <= st <= 99999 and val is not None and val>=0 and win is not None and abs(win)==60):
            continue
        if st in result and result[st]!=val:
            excluded.add(st)
        result[st]=val
    return {k:v for k,v in result.items() if k not in excluded}


def parse_station_info(raw):
    result={}
    excluded=set()
    saw_header=False
    for line in checked_text(raw).splitlines():
        line=line.strip()
        if not line:
            continue
        t=tokens(line.lstrip('#'))
        if [s.upper() for s in t[:3]] in (['STN_ID','LON','LAT'],['STN','LON','LAT']):
            saw_header=True
            continue
        if line.startswith('#') or len(t)<3 or not t[0].isdigit():
            continue
        # KMA's documented position of the leading station ID, longitude and latitude.
        st=int(t[0])
        lon,lat=map(number,t[1:3])
        if not (1<=st<=99999 and lon is not None and lat is not None and
              124.0<=lon<=132.1 and 32.5<=lat<=39.0):
            continue
        point=Station(st,lon,lat)
        if st in result and result[st]!=point:
            excluded.add(st)
        result[st]=point
    if not result and not saw_header:
        raise WeatherUnavailable('기상청 관측소 좌표 응답을 확인하지 못했습니다.')
    return {k:v for k,v in result.items() if k not in excluded}


def accumulator(frames,stations):
    """12 separate 60-minute totals, descending by time, from the same gauge."""
    if len(frames)!=12 or not all(frames):
        return ()
    common=set(stations).intersection(*(set(f) for f in frames))
    accepted=[]
    for st in sorted(common):
        v=[f[st] for f in frames]
        if all(isfinite(r) and r>=0 for r in v):
            accepted.append((stations[st],round(sum(v[:3]),5),round(sum(v),5)))
    return tuple(accepted)


class KmaAws:
    """Shared, bounded process cache. Initial load fetches 12 frames; subsequent refreshes reuse settled frames."""
    def __init__(self,api_key,*,transport=None,max_workers=4):
        self._key=api_key.strip()
        if not self._key:
            raise WeatherUnavailable('서버에 기상청 API허브 키가 연결되지 않았습니다.')
        self._transport=transport or self._download
        self._frame=OrderedDict()
        self._stations=OrderedDict()
        self._flight=Lock()
        self._workers=max_workers
        self._cooldown_until=0.0

    def _download(self,base,params):
        query=urlencode({**params,'authKey':self._key})
        try:
            req=Request(f'{base}?{query}',headers={'User-Agent':'KoreaFloodStudy/1.0'})
            with urlopen(req,timeout=8) as response:
                return response.read(MAX_RESPONSE+1)
        except HTTPError as exc:
            if exc.code in (400,401,403,429):
                raise WeatherUnavailable('AWS API 요청이 거절됐습니다. 키, 활용 승인 또는 호출 한도를 확인하세요.') from None
            raise WeatherUnavailable('기상청 서버에서 관측 응답을 받지 못했습니다.') from None
        except (URLError, OSError, TimeoutError, ValueError):
            raise WeatherUnavailable('기상청 서버에 연결하지 못했습니다.') from None

    def _fresh_frame(self,obs):
        try:
            raw=self._transport(AWS_HOURLY,dict(var='RN',tm=tm(obs),stn='0',help='1'))
            return parse_rn(raw,obs),None
        except WeatherUnavailable as exc:
            return {},str(exc)

    def _coordinates(self,end):
        day=end.date()
        cached=self._stations.get(day)
        if cached is not None:
            self._stations.move_to_end(day)
            return cached
        sites={}
        errors=[]
        for kind in ('AWS','SFC'):
            try:
                raw=self._transport(KMA_STATIONS,dict(inf=kind,stn='0',tm=tm(end),help='1'))
                parsed=parse_station_info(raw)
                if kind=='AWS':
                    sites.update(parsed)
                else:
                    # Prefer the AWS location for IDs in its response. The SFC registry fills ASOS IDs.
                    sites={**parsed,**sites}
            except WeatherUnavailable as exc:
                errors.append(str(exc))
        if not sites:
            raise WeatherUnavailable(errors[0] if errors else '관측소 좌표를 확보하지 못했습니다.')
        self._stations[day]=sites
        while len(self._stations)>10:
            self._stations.popitem(last=False)
        return sites

    def _get_at(self,end):
        if time.monotonic()<self._cooldown_until:
            return Rain(end,PUBLIC_SOURCE,(), 'API 요청이 거절되어 잠시 재요청을 쉬고 있습니다.')
        issued=[end-i*HOUR for i in range(12)]
        stamp=time.monotonic()
        available={}
        missing=[]
        for t in issued:
            cached=self._frame.get(t)
            if cached is None or cached[0]<=stamp:
                missing.append(t)
            else:
                self._frame.move_to_end(t)
                available[t]=(cached[1],cached[2])
        if missing:
            # Probe first, to avoid launching many calls on a rejected key.
            seed=missing.pop(0)
            v,error=self._fresh_frame(seed)
            available[seed]=(v,error)
            if not v:
                if error and '거절' in error:
                    self._cooldown_until=stamp+120
                self._frame[seed]=(stamp+150,v,error)
                while len(self._frame)>108:
                    self._frame.popitem(last=False)
                return Rain(end,PUBLIC_SOURCE,(),error or '선택 시각의 AWS 시간 강수 자료가 아직 없거나 제공 범위 밖입니다.')
            with ThreadPoolExecutor(max_workers=self._workers) as pool:
                for t,values in zip(missing,pool.map(self._fresh_frame,missing)):
                    available[t]=values
            for t in issued:
                if t in self._frame and self._frame[t][0]>stamp:
                    continue
                vals,err=available[t]
                ttl=(3600 if (end-t)<=2*HOUR else 20*3600) if vals else 150
                self._frame[t]=(stamp+ttl,vals,err)
            while len(self._frame)>108:
                self._frame.popitem(last=False)
        inputs=[available[t][0] for t in issued]
        if not all(inputs):
            error=next((available[t][1] for t in issued if available[t][1]),None)
            return Rain(end,PUBLIC_SOURCE,(),error or '최근 12시간에 빠진 AWS 자료가 있습니다.')
        try:
            coords=self._coordinates(end)
        except WeatherUnavailable as exc:
            return Rain(end,PUBLIC_SOURCE,(),str(exc))
        observations=accumulator(inputs,coords)
        return Rain(end,PUBLIC_SOURCE,observations,
                    None if observations else '좌표와 연속 12시간 강수가 모두 확인된 관측소가 없습니다.')

    def snapshot(self,target,*,live_now=None):
        obs=hourstamp(target)
        if live_now is not None and live_now.astimezone(KST)<obs:
            raise ValueError('미래 시각은 관측으로 조회할 수 없습니다.')
        with self._flight:
            f=self._get_at(obs)
            if live_now is None or f.readings or (f.reason and '거절' in f.reason):
                return f
            earlier=obs-HOUR
            if live_now.astimezone(KST)-earlier>timedelta(minutes=MAX_LIVE_MINUTES):
                return f
            old=self._get_at(earlier)
            if old.readings:
                return Rain(earlier,PUBLIC_SOURCE,old.readings,'최신 정시 자료가 지연되어 직전 자료를 사용했습니다.')
            return f
