from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import neurokit2 as nk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import (
    EcgSegment,
    EcgFeatures,
    QualityInfo,
    RPeaksInfo,
    HrvTimeInfo,
    PomodoroWorkRequest,
    PomodoroWorkSummary,
    TrendPoint,
    RrSummary,
    HrSummary,
)

APP_ORIGINS = ["http://localhost:3000"]

# Pragmatic robustness thresholds (not medical rules).
MIN_SAMPLES_FOR_PROCESS = 2_000
MIN_RPEAKS_FOR_HRV = 3

app = FastAPI(title="ECG Service", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=APP_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass(frozen=True)
class _QualityReport:
    signal_ok: bool
    missing_ratio: float
    quality_index_mean: float
    notes: List[str]


def _to_2d_array(samples: List[List[float]]) -> np.ndarray:
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2:
        raise ValueError("samples must be a 2D array with shape [N, C].")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("samples is empty.")
    return arr


def _select_primary_channel(samples_2d: np.ndarray, channel_index: int = 0) -> np.ndarray:
    if channel_index < 0 or channel_index >= samples_2d.shape[1]:
        raise ValueError(f"Invalid channel_index={channel_index}.")
    return samples_2d[:, channel_index]


def _missing_ratio(x: np.ndarray) -> float:
    if x.size == 0:
        return 1.0
    bad = ~np.isfinite(x)
    return float(np.sum(bad)) / float(x.size)


def _clean_ecg(x: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        idx = np.arange(x.size)
        good = np.isfinite(x)
        if np.sum(good) < 2:
            raise ValueError("Too many non-finite samples to interpolate.")
        x = np.interp(idx, idx[good], x[good])
    return nk.ecg_clean(x, sampling_rate=sampling_rate_hz)


def _compute_quality(ecg_cleaned: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
    return nk.ecg_quality(ecg_cleaned, sampling_rate=sampling_rate_hz)


def _detect_rpeaks(ecg_cleaned: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate_hz)
    return np.asarray(rpeaks["ECG_R_Peaks"], dtype=int)


def _rr_intervals_ms(rpeaks_idx: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
    if rpeaks_idx.size < 2:
        return np.asarray([], dtype=float)
    rr_samples = np.diff(rpeaks_idx).astype(float)
    return rr_samples / float(sampling_rate_hz) * 1000.0


def _hrv_from_rr(rr_ms: np.ndarray) -> HrvTimeInfo:
    if rr_ms.size < 2:
        return HrvTimeInfo(mean_hr_bpm=0.0, rmssd_ms=0.0, sdnn_ms=0.0)

    mean_rr = float(np.mean(rr_ms))
    mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0

    sdnn = float(np.std(rr_ms, ddof=0))
    diff_rr = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diff_rr**2))) if diff_rr.size else 0.0

    return HrvTimeInfo(mean_hr_bpm=round(mean_hr, 2), rmssd_ms=round(rmssd, 2), sdnn_ms=round(sdnn, 2))


def _rr_summary(rr_ms: np.ndarray) -> RrSummary:
    if rr_ms.size == 0:
        return RrSummary()

    med = float(np.median(rr_ms))
    if med <= 0:
        outlier_ratio = 0.0
    else:
        outlier_ratio = float(np.mean(np.abs(rr_ms - med) / med > 0.2))

    return RrSummary(
        n=int(rr_ms.size),
        mean_ms=round(float(np.mean(rr_ms)), 2),
        std_ms=round(float(np.std(rr_ms, ddof=0)), 2),
        min_ms=round(float(np.min(rr_ms)), 2),
        max_ms=round(float(np.max(rr_ms)), 2),
        outlier_ratio=round(outlier_ratio, 3),
    )


def _process_segment(segment: EcgSegment) -> Tuple[_QualityReport, RPeaksInfo, HrvTimeInfo, np.ndarray]:
    samples_2d = _to_2d_array(segment.samples)
    ecg_raw = _select_primary_channel(samples_2d, channel_index=0)

    notes: List[str] = []
    miss_ratio = _missing_ratio(ecg_raw)

    if miss_ratio > 0.05:
        notes.append(f"High missing_ratio={miss_ratio:.3f} (auto-interpolation applied).")

    if ecg_raw.size < MIN_SAMPLES_FOR_PROCESS:
        notes.append(f"Too short segment: n_samples={ecg_raw.size}.")
        return (
            _QualityReport(False, miss_ratio, 0.0, notes),
            RPeaksInfo(indices=[], method="neurokit2"),
            HrvTimeInfo(),
            np.asarray([], dtype=float),
        )

    ecg_cleaned = _clean_ecg(ecg_raw, segment.sampling_rate_hz)

    # Quality (optional)
    q_mean = 0.0
    try:
        q = _compute_quality(ecg_cleaned, segment.sampling_rate_hz)
        q_mean = float(np.nanmean(q)) if q.size else 0.0
    except Exception as exc:
        notes.append(f"ecg_quality failed: {exc}")
        q_mean = 0.0

    # R-peaks (required for HR/HRV)
    try:
        peaks = _detect_rpeaks(ecg_cleaned, segment.sampling_rate_hz)
    except Exception as exc:
        notes.append(f"R-peak detection failed: {exc}")
        return (
            _QualityReport(False, miss_ratio, q_mean, notes),
            RPeaksInfo(indices=[], method="neurokit2"),
            HrvTimeInfo(),
            np.asarray([], dtype=float),
        )

    rr_ms = _rr_intervals_ms(peaks, segment.sampling_rate_hz)

    # HRV time
    if peaks.size < MIN_RPEAKS_FOR_HRV:
        notes.append(f"Not enough R-peaks for HRV: n_peaks={int(peaks.size)}.")
        signal_ok = False
        hrv = HrvTimeInfo()
    else:
        # Prefer RR-based stable calculation for summary usage
        hrv = _hrv_from_rr(rr_ms)
        signal_ok = True

    return (
        _QualityReport(signal_ok, miss_ratio, round(q_mean, 4), notes if notes else ["ok"]),
        RPeaksInfo(indices=peaks.tolist(), method="neurokit2"),
        hrv,
        rr_ms,
    )


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.post("/ecg/features", response_model=EcgFeatures)
def ecg_features(segment: EcgSegment) -> EcgFeatures:
    q, rpeaks, hrv, _ = _process_segment(segment)
    return EcgFeatures(
        segment_id=segment.segment_id,
        quality=QualityInfo(
            signal_ok=q.signal_ok,
            missing_ratio=round(q.missing_ratio, 6),
            quality_index_mean=round(q.quality_index_mean, 4),
            notes=q.notes,
        ),
        rpeaks=rpeaks,
        hrv_time=hrv,
    )


@app.post("/ecg/pomodoro/end", response_model=PomodoroWorkSummary)
def end_pomodoro(req: PomodoroWorkRequest) -> PomodoroWorkSummary:
    # Ensure chronological order for trend & RR concatenation
    segments = sorted(req.segments, key=lambda s: s.start_time_unix_ms)

    rr_all: List[float] = []
    q_means: List[float] = []
    miss_ratios: List[float] = []
    signal_ok_count = 0
    notes: List[str] = []
    trend: List[TrendPoint] = []

    for seg in segments:
        q, _, hrv, rr_ms = _process_segment(seg)

        rr_all.extend(rr_ms.tolist())
        q_means.append(float(q.quality_index_mean))
        miss_ratios.append(float(q.missing_ratio))
        signal_ok_count += int(q.signal_ok)
        notes.extend(q.notes)

        t_offset_s = int(max(0, (seg.start_time_unix_ms - req.work_start_unix_ms) // 1000))
        trend.append(
            TrendPoint(
                t_offset_s=t_offset_s,
                mean_hr_bpm=hrv.mean_hr_bpm,
                rmssd_ms=hrv.rmssd_ms,
                sdnn_ms=hrv.sdnn_ms,
                quality_index_mean=float(q.quality_index_mean),
                signal_ok=q.signal_ok,
            )
        )

    rr_all_arr = np.asarray(rr_all, dtype=float)

    hrv_total = _hrv_from_rr(rr_all_arr)
    rr_sum = _rr_summary(rr_all_arr)

    # HR summary derived from RR distribution (per-beat HR)
    if rr_all_arr.size > 0:
        hr_inst = 60000.0 / rr_all_arr
        hr_sum = HrSummary(
            mean_bpm=round(float(np.mean(hr_inst)), 2),
            min_bpm=round(float(np.min(hr_inst)), 2),
            max_bpm=round(float(np.max(hr_inst)), 2),
        )
    else:
        hr_sum = HrSummary()

    # Aggregate quality
    n_seg = max(1, len(segments))
    q_mean_total = float(np.mean(q_means)) if q_means else 0.0
    miss_mean_total = float(np.mean(miss_ratios)) if miss_ratios else 1.0
    signal_ok_ratio = signal_ok_count / float(n_seg)

    # Define overall signal_ok conservatively
    overall_ok = bool(signal_ok_ratio >= 0.7 and rr_all_arr.size >= 10)

    # De-duplicate notes (keep order)
    seen = set()
    uniq_notes = []
    for n in notes:
        if n not in seen:
            uniq_notes.append(n)
            seen.add(n)

    return PomodoroWorkSummary(
        user_id=req.user_id,
        session_id=req.session_id,
        work_start_unix_ms=req.work_start_unix_ms,
        work_end_unix_ms=req.work_end_unix_ms,
        duration_s=int(max(0, (req.work_end_unix_ms - req.work_start_unix_ms) // 1000)),
        quality=QualityInfo(
            signal_ok=overall_ok,
            missing_ratio=round(miss_mean_total, 6),
            quality_index_mean=round(q_mean_total, 4),
            notes=uniq_notes if uniq_notes else ["ok"],
        ),
        hr_summary=hr_sum,
        hrv_time=hrv_total,
        rr_summary=rr_sum,
        trend_1min=trend,
    )