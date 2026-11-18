"""Read the project's tables without pyarrow's Parquet DLL.

Each artifacts/*.parquet has a sibling *.npz holding the same columns (made from the
parquet file whose SHA-256 is stored inside). Windows application-control policies
(Smart App Control) can block pyarrow's Parquet module, so the .npz copy is read first.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _from_npz(npz_path, parquet_path):
    with np.load(npz_path, allow_pickle=False) as z:
        if parquet_path.exists() and str(z["__source_sha256__"]) != _sha(parquet_path):
            raise ValueError(f"{npz_path.name}가 {parquet_path.name}와 다른 버전입니다.")
        data = {}
        for name in z["__columns__"].tolist():
            v = z["col_" + name]
            if "tz_" + name in z.files:
                data[name] = pd.to_datetime(v, unit="ns", utc=True).tz_convert(str(z["tz_" + name]))
            elif "nullableint_" + name in z.files:
                data[name] = pd.array(v, dtype="Int64")
            elif v.dtype.kind == "U":
                data[name] = v.astype(object)
            else:
                data[name] = v
    return pd.DataFrame(data)


def read_table(parquet_path):
    """Same DataFrame as pandas.read_parquet(parquet_path)."""
    parquet_path = Path(parquet_path)
    npz_path = parquet_path.with_suffix(".npz")
    if npz_path.exists():
        return _from_npz(npz_path, parquet_path)
    return pd.read_parquet(parquet_path)
