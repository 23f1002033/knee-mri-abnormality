import os
import shutil
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

D, S = 32, 256
MM_FOV = 140.0
VENDORS = ["siemens", "ge", "philips", "toshiba", "canon", "hitachi"]
GEO_VENDORS = {"siemens", "philips"}
PLANE_KEYS = {"Sagittal": "sag", "Coronal": "cor", "Axial": "axi"}


def vendor(m):
    m = str(m).lower()
    for v in VENDORS:
        if m.startswith(v):
            return v
    return "other"


def axis_name(v):
    i = int(np.argmax(np.abs(v)))
    return ["RL", "AP", "IS"][i][int(v[i] > 0)]


def scan_series(series_dir):
    entries = sorted(os.scandir(series_dir), key=lambda e: e.name)
    ds = pydicom.dcmread(entries[len(entries) // 2].path, stop_before_pixels=True)
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    out = {
        "n_files": len(entries),
        "rows": int(ds.Rows),
        "thick_mm": float(ds.SliceThickness) if "SliceThickness" in ds else np.nan,
        "vendor": vendor(ds.get("Manufacturer")),
        "row_dir": axis_name(iop[:3]),
        "col_dir": axis_name(iop[3:]),
        "center_x": np.nan,
    }
    tag = str(ds.get("Laterality") or ds.get("ImageLaterality") or "").strip().upper()
    out["lat_tag"] = {"LEFT": "L", "RIGHT": "R"}.get(tag, tag)
    ipp, ps = ds.get("ImagePositionPatient"), ds.get("PixelSpacing")
    if ipp is not None and ps is not None:
        c = (np.array(ipp, dtype=float) + iop[:3] * ds.Columns / 2 * float(ps[1])
             + iop[3:] * ds.Rows / 2 * float(ps[0]))
        out["center_x"] = float(c[0])
    return out


def _scan_one(r, root, split):
    d = os.path.join(root, split, r.StudyInstanceUID, r.SeriesInstanceUID)
    try:
        return {"SeriesInstanceUID": r.SeriesInstanceUID, "error": None, **scan_series(d)}
    except Exception as err:
        return {"SeriesInstanceUID": r.SeriesInstanceUID, "error": repr(err)[:200]}


def scan_all(series_df, root, split, workers=16):
    fn = partial(_scan_one, root=root, split=split)
    with ThreadPoolExecutor(workers) as ex:
        info = list(ex.map(fn, series_df.itertuples(index=False)))
    return series_df.merge(pd.DataFrame(info), on="SeriesInstanceUID", how="left")


def rank_series(inv):
    return (inv.assign(ok=inv["error"].isna(), thick_gap=(inv["thick_mm"] - 3).abs())
               .sort_values(["ok", "Fluid_Sensitive", "thick_gap", "rows"],
                            ascending=[False, False, True, False])
               .drop(columns=["ok", "thick_gap"]))


def select_series(inv):
    return (rank_series(inv)
               .drop_duplicates(["StudyInstanceUID", "Anatomical_Plane"])
               .sort_values(["StudyInstanceUID", "Anatomical_Plane"])
               .reset_index(drop=True))


def _single_tag(s):
    t = set(s.dropna()) & {"L", "R"}
    return t.pop() if len(t) == 1 else None


def study_laterality(inv):
    tag = inv.groupby("StudyInstanceUID")["lat_tag"].agg(_single_tag).rename("tag")
    cor = (inv[inv["Anatomical_Plane"] == "Coronal"].groupby("StudyInstanceUID")
              .agg(center_x=("center_x", "median"), vendor=("vendor", "first")))
    lat = cor.join(tag, how="outer")
    geo_ok = lat["vendor"].isin(GEO_VENDORS) & (lat["center_x"].abs() > 30)
    lat["geo"] = pd.Series(np.where(lat["center_x"] > 0, "L", "R"), index=lat.index).where(geo_ok)
    lat["side"] = lat["tag"].fillna(lat["geo"])
    lat["side_src"] = np.select([lat["tag"].notna(), lat["geo"].notna()], ["tag", "geo"], "unknown")
    return lat.reset_index()


def load_volume(series_dir):
    dss = [pydicom.dcmread(e.path) for e in os.scandir(series_dir)]
    qa = {"n": len(dss), "sort": "position"}
    iop = np.array(dss[0].ImageOrientationPatient, dtype=float)
    normal = np.cross(iop[:3], iop[3:])
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    try:
        pos = np.array([normal @ np.array(ds.ImagePositionPatient, dtype=float) for ds in dss])
    except AttributeError:
        pos = np.array([float(ds.InstanceNumber) for ds in dss])
        qa["sort"] = "instance"
    order = np.argsort(pos, kind="stable")
    gaps = np.diff(pos[order])
    med = float(np.median(gaps)) if len(gaps) else 0.0
    qa["spacing"] = round(med, 3)
    qa["dup_pos"] = int((gaps < 1e-3).sum())
    qa["big_gaps"] = int((gaps > 1.5 * med).sum()) if med > 0 else 0
    ps = dss[order[0]].get("PixelSpacing")
    qa["px"] = (float(ps[0]), float(ps[1])) if ps is not None else None
    arrs = [dss[i].pixel_array.astype(np.float32) for i in order]
    qa["mixed_shapes"] = len({a.shape for a in arrs}) > 1
    return (None if qa["mixed_shapes"] else np.stack(arrs)), qa


def normalize(vol):
    lo, hi = np.percentile(vol, [0.5, 99.5])
    return np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def pad_square(vol):
    d, h, w = vol.shape
    m = max(h, w)
    out = np.zeros((d, m, m), dtype=vol.dtype)
    y, x = (m - h) // 2, (m - w) // 2
    out[:, y:y + h, x:x + w] = vol
    return out


def resample(vol):
    t = torch.from_numpy(pad_square(vol))[:, None]
    t = F.interpolate(t, size=(S, S), mode="bilinear", antialias=True, align_corners=False)[:, 0]
    t = t.reshape(t.shape[0], S * S).T[None]
    t = F.interpolate(t, size=D, mode="linear", align_corners=False)
    return (t[0].T.reshape(D, S, S).clamp(0, 1) * 255).round().to(torch.uint8).numpy()


def leg_mask(img, thr=0.1):
    from scipy import ndimage
    m = img > thr
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == int(np.argmax(sizes)) + 1
    return m


def knee_crop(vol, size, thr=0.1):
    d, h, w = vol.shape
    ys, xs = np.nonzero(leg_mask(vol.mean(0), thr))
    if len(ys) > 100:
        cy, cx = int(ys.mean()), int(xs.mean())
    else:
        cy, cx = h // 2, w // 2
    out = np.zeros((d, size, size), vol.dtype)
    y0, x0 = cy - size // 2, cx - size // 2
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y0 + size, h), min(x0 + size, w)
    out[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = vol[:, sy0:sy1, sx0:sx1]
    return out, (cy, cx)


def resample_mm(vol, px):
    t_mm = MM_FOV / S
    d, h, w = vol.shape
    nh = max(1, int(round(h * px[0] / t_mm)))
    nw = max(1, int(round(w * px[1] / t_mm)))
    t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))[:, None]
    t = F.interpolate(t, size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)[:, 0].numpy()
    crop, _ = knee_crop(t, S)
    t = torch.from_numpy(crop).reshape(d, S * S).T[None]
    t = F.interpolate(t, size=D, mode="linear", align_corners=False)
    return (t[0].T.reshape(D, S, S).clamp(0, 1) * 255).round().to(torch.uint8).numpy()


def make_volume(vol, qa, geometry="px"):
    if geometry == "mm" and qa.get("px"):
        return resample_mm(normalize(vol), qa["px"])
    return resample(normalize(vol))


def process_study(rows, root, split, geometry="px"):
    vols, qas = {}, []
    for r in rows.itertuples(index=False):
        d = os.path.join(root, split, r.StudyInstanceUID, r.SeriesInstanceUID)
        vol, qa = load_volume(d)
        key = PLANE_KEYS[r.Anatomical_Plane]
        qa.update(StudyInstanceUID=r.StudyInstanceUID, plane=key)
        if vol is not None:
            vols[key] = make_volume(vol, qa, geometry)
        qas.append(qa)
    return vols, qas


def _init_worker():
    torch.set_num_threads(1)


def _cache_one(item, root, split, out_dir, min_free_gb, geometry="px"):
    sid, rows = item
    path = os.path.join(out_dir, f"{sid}.npz")
    if os.path.exists(path):
        return []
    if shutil.disk_usage(out_dir).free < min_free_gb * 1e9:
        return [{"StudyInstanceUID": sid, "error": "low disk"}]
    try:
        vols, qas = process_study(rows, root, split, geometry)
        np.savez_compressed(path, **vols)
        return [{**q, "error": None} for q in qas]
    except Exception as err:
        return [{"StudyInstanceUID": sid, "error": repr(err)[:200]}]


def build_cache(sel, root, split, out_dir, workers=4, min_free_gb=1.0, geometry="px"):
    os.makedirs(out_dir, exist_ok=True)
    groups = list(sel.groupby("StudyInstanceUID"))
    fn = partial(_cache_one, root=root, split=split, out_dir=out_dir, min_free_gb=min_free_gb,
                 geometry=geometry)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"), initializer=_init_worker) as ex:
        res = list(tqdm(ex.map(fn, groups, chunksize=4), total=len(groups), mininterval=60))
    return pd.DataFrame([q for qs in res for q in qs])


def load_study(inv_study, root, split, geometry="px"):
    vols, notes = {}, []
    for plane, cands in rank_series(inv_study).groupby("Anatomical_Plane", sort=False):
        key = PLANE_KEYS.get(plane)
        if key is None:
            continue
        for r in cands.itertuples(index=False):
            d = os.path.join(root, split, r.StudyInstanceUID, r.SeriesInstanceUID)
            try:
                vol, qa = load_volume(d)
                if vol is None:
                    raise ValueError("mixed slice shapes")
                vols[key] = make_volume(vol, qa, geometry)
                break
            except Exception as err:
                notes.append(f"{key} {r.SeriesInstanceUID[-6:]}: {type(err).__name__}")
    return vols, notes


def _study_one(item, root, split, geometry="px"):
    sid, inv_study = item
    t0 = time.time()
    try:
        vols, notes = load_study(inv_study, root, split, geometry)
    except Exception as err:
        vols, notes = {}, [f"study: {type(err).__name__}"]
    return sid, vols, notes, time.time() - t0


def iter_studies(inv, root, split, workers=4, geometry="px"):
    groups = list(inv.groupby("StudyInstanceUID"))
    fn = partial(_study_one, root=root, split=split, geometry=geometry)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"), initializer=_init_worker) as ex:
        yield from ex.map(fn, groups, chunksize=2)
