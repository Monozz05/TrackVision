"""The Test Videos section (decision 0017): the working, shown on street videos that have no ground truth.

YOLO11 finds people, the existing IoU tracker gives them IDs, the frozen CV and CA forecast from those tracks, and the
frozen injectors act on the same tracks. None of that is ground truth, so no forecast is ever compared with a true
position: no ADE, FDE or other accuracy value exists in this section. The only numbers are counts, and how far errors
moved a forecast, which is change and not accuracy.

A forecast reads a person's position as the mean of their foot point over the step before it. A box edge jumps from frame
to frame, which CV and CA would read as speed, and the mean takes nothing from the future. A person with no box in a step
has no point there, and a forecast needs `history_len` points in a row.

These videos are not a dataset and not experiment P-1 or P-9. Their browser copies and boxes live apart from every
dataset's, so nothing here can reach the Workspace, and results stay in memory and never reach `runs/`.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trackvision import config
from trackvision.platform.config import Settings
from trackvision.platform.services import video_tracks as vt
from trackvision.platform.services.result_reader import ArtifactMalformed, ResultReader
from trackvision.platform.services.result_views import ViewUnavailable
from trackvision.platform.services.workspace import (DETECTIONS_SCHEMA, MAX_SIGMA_PX, MOVING_CAUTION, OVERLAY_SEED_INDEX, Base,
                                                     Loaded, SceneUnavailable, _achieved, _in_pixels, _lost, _overlay)
from trackvision.synthetic.tracks import GridTrack

KEPT_RESULTS = 16
PREDICTORS = ("CV", "CA")
LABEL = ("Display only. These videos are not a dataset and not a research result. Boxes come from YOLO11 and IDs from the IoU "
         "tracker, so they can be wrong, and one person can get several IDs. Positions are in video pixels.")
NO_ACCURACY = "No true positions exist for this video, so ADE and FDE cannot be measured."
MOVED_NOTE = "Forecast moved is the mean distance between the forecast with and without errors: change, not accuracy."
TRACK_NOTE = "Tracker output for display. A forecast needs 2.8 s of unbroken track, so only some people have one."
NOT_READY = "This video has not been prepared. Run scripts/prepare_test_videos.py, then scripts/yolo_detect_test_videos.py."


@dataclass(frozen=True)
class StreetVideo:
    """One video on the page. This list is also the allowlist: no other name can reach the disk."""
    id: str
    title: str
    file: str                        # the original, in `videos/`
    sha1: str                        # as published by Wikimedia Commons
    author: str
    licence: str
    licence_url: str
    source_url: str
    camera: str
    caution: str | None = None


STREET_VIDEOS = (
    StreetVideo("buchanan-street", "Buchanan Street, Glasgow", "Buchanan Street, Glasgow.webm",
                "c22e1c88b31fdebe13d22c6f7b61c03bc00033f1", "Learning, Teaching and Web Services, University of Edinburgh",
                "CC BY-SA 3.0", "https://creativecommons.org/licenses/by-sa/3.0",
                "https://commons.wikimedia.org/wiki/File:Pedestrians_in_Buchanan_Street,_Glasgow.webm", "Moves slightly",
                MOVING_CAUTION),
    StreetVideo("waiting-to-cross", "People waiting to cross", "People waiting to cross.webm",
                "635373fdb0ccb21b9bd716666893faee18598669", "Amada44", "CC BY-SA 3.0",
                "https://creativecommons.org/licenses/by-sa/3.0",
                "https://commons.wikimedia.org/wiki/File:People_waiting_to_cross_the_street.webm", "Almost still"),
    StreetVideo("singapore-scramble", "Scramble crossing, Singapore", "Scramble crossing, Singapore.webm",
                "2374de686ca82658c9c290e156f2dce2506f1aa2", "Jun Jie Yam", "CC BY 4.0",
                "https://creativecommons.org/licenses/by/4.0",
                "https://commons.wikimedia.org/wiki/File:Scramble_Crossing_at_Robinson_Road_in_Singapore_-_September_2022.webm",
                "Moves, filmed while walking", MOVING_CAUTION),
    StreetVideo("toronto-diagonal", "Diagonal crossing, Toronto", "Diagonal crossing, Toronto.webm",
                "96fe4eb1bf26928aff17c9df771d0ffd0bf55aad", "Raysonho", "CC0",
                "http://creativecommons.org/publicdomain/zero/1.0/deed.en",
                "https://commons.wikimedia.org/wiki/File:DiagonalCrosswalkYongeDundas.webm", "Fixed"),
    StreetVideo("caracas-boulevard", "Sabana Grande boulevard, Caracas", "Sabana Grande boulevard, Caracas.webm",
                "020c7bf941fed3bb47d6a8dafd2ef89fc8c740ff", "Vicente Quintero (QuinteroP)", "CC BY 3.0",
                "https://creativecommons.org/licenses/by/3.0",
                "https://commons.wikimedia.org/wiki/File:Sabana_Grande_Caracas._People_walking_on_the_Boulevard_of_Sabana_Grande,"
                "_famous_in_Caracas,_Venezuela.webm", "Fixed"),
)
VIDEOS = {video.id: video for video in STREET_VIDEOS}


def sampled(frames, uv, boxes, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One point per step: grid frames, the mean position over the step ending there, and the last box in that step."""
    frames = np.asarray(frames, dtype=np.int64)
    uv, boxes = np.asarray(uv, dtype=np.float64), np.asarray(boxes, dtype=np.float64)
    cells, first = np.unique((frames + step - 1) // step, return_index=True)
    stops = np.r_[first[1:], len(frames)]
    keep = cells * step <= frames[-1]                       # a person last seen part-way through a step has no point at its end
    points = np.array([uv[a:b].mean(axis=0) for a, b in zip(first[keep], stops[keep])]).reshape(-1, 2)
    return cells[keep] * step, points, boxes[stops[keep] - 1].reshape(-1, 4)


@dataclass(frozen=True)
class HistoryWindows:
    """Past points only. With no ground truth a window has no future, which is all the injectors and predictors read."""
    track_ids: np.ndarray            # (N,)
    frames: np.ndarray               # (N, history_len)
    observed: np.ndarray             # (N, history_len, 2)
    history_len: int

    def __len__(self) -> int:
        return len(self.track_ids)


@dataclass(frozen=True)
class Histories:
    windows: HistoryWindows


def history_windows(tracks, history_len: int, step: int) -> HistoryWindows:
    """Every run of `history_len` points in a row, for every person."""
    ids, frames, observed = [], [], []
    for track in tracks:
        for k in range(history_len - 1, len(track.frames)):
            start = k - history_len + 1
            if track.frames[k] - track.frames[start] == (history_len - 1) * step:
                ids.append(track.ped_id)
                frames.append(track.frames[start:k + 1])
                observed.append(track.positions[start:k + 1])
    return HistoryWindows(np.asarray(ids, dtype=np.int64), np.asarray(frames, dtype=np.int64).reshape(-1, history_len),
                          np.asarray(observed, dtype=np.float64).reshape(-1, history_len, 2), history_len)


@dataclass(frozen=True)
class Prepared:
    """A video ready for forecasts. It stays unchanged for the life of the process."""
    loaded: Loaded
    clean: dict                      # predictor name -> (N, horizon_len, 2) forecasts with no errors
    rows: list                       # the same people on the 0.4 s grid, in the shape the page draws
    seconds_per_step: float


class StreetVideos:
    def __init__(self, settings: Settings, reader: ResultReader):
        self.copies_dir = settings.cache_dir.parent / "test_videos"
        self.detections_dir = settings.cache_dir.parent / "test_video_detections"
        self._reader = reader
        self._tracks: dict[tuple, dict] = {}
        self._prepared: dict[tuple, Prepared] = {}
        self._results: OrderedDict[tuple, dict] = OrderedDict()
        self._lock = threading.Lock()

    def _config(self):
        return config.validate().config

    def _video(self, video_id: str) -> StreetVideo:
        video = VIDEOS.get(video_id)
        if video is None:
            raise ViewUnavailable(f"there is no test video called {video_id}")
        return video

    def _paths(self, video: StreetVideo) -> tuple[Path, Path, Path]:
        return (self.copies_dir / f"{video.id}.mp4", self.copies_dir / f"{video.id}.json",
                self.detections_dir / f"{video.id}.json")

    def _meta(self, video: StreetVideo) -> dict | None:
        copy, meta, detections = self._paths(video)
        if not (copy.is_file() and meta.is_file() and detections.is_file()):
            return None
        try:
            return json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _ready(self, video_id: str) -> tuple[StreetVideo, dict]:
        video = self._video(video_id)
        meta = self._meta(video)
        if meta is None:
            raise SceneUnavailable(NOT_READY)
        return video, meta

    def catalogue(self) -> dict:
        items = []
        for video in STREET_VIDEOS:
            meta = self._meta(video)
            items.append({"id": video.id, "title": video.title, "author": video.author, "licence": video.licence,
                          "licence_url": video.licence_url, "source_url": video.source_url, "camera": video.camera,
                          "caution": video.caution, "ready": meta is not None, "reason": None if meta else NOT_READY,
                          "width": (meta or {}).get("width"), "height": (meta or {}).get("height"),
                          "fps": (meta or {}).get("fps"), "frames": (meta or {}).get("frames")})
        return {"items": items, "predictors": list(PREDICTORS), "label": LABEL, "no_accuracy": NO_ACCURACY,
                "moved_note": MOVED_NOTE,
                "controls": {"A": {"max": MAX_SIGMA_PX, "unit": "px", "protocol_max": None},
                             **{code: {"max": 1.0, "unit": "fraction", "protocol_max": None} for code in "BCD"}}}

    def video_file(self, video_id: str) -> Path:
        video, _ = self._ready(video_id)
        return self._paths(video)[0]

    def _detections(self, video: StreetVideo) -> dict:
        document = self._reader.read_json(self._paths(video)[2], cache=False)
        frames = document.get("by_frame")
        if document.get("schema") != DETECTIONS_SCHEMA or document.get("scene") != video.id or not isinstance(frames, list) \
                or len(frames) != document.get("frames"):
            raise ArtifactMalformed(f"{video.id}.json is not a valid detection file")
        return document

    def tracks(self, video_id: str) -> dict:
        """Everyone the tracker follows, frame by frame, with its existing default settings."""
        video, _ = self._ready(video_id)
        key = (video.id, self._paths(video)[2].stat().st_mtime_ns)
        with self._lock:
            if key in self._tracks:
                return self._tracks[key]
        document = self._detections(video)
        protocol = self._config().frozen["protocol"]["static"]
        out = {"scene": video.id, "source": f"IoU tracker on {document.get('model')} detections", "note": TRACK_NOTE,
               "frame_step": 1, "history_len": protocol["history_len"], "horizon_len": protocol["horizon_len"],
               "tracks": vt.video_tracks(document)}
        with self._lock:
            self._tracks[key] = out
        return out

    def _prepare(self, video_id: str) -> Prepared:
        from trackvision.experiments import evaluation
        video, meta = self._ready(video_id)
        key = (video.id, self._paths(video)[2].stat().st_mtime_ns)
        with self._lock:
            if key in self._prepared:
                return self._prepared[key]
        cfg = self._config()
        protocol = cfg.frozen["protocol"]["static"]
        step = max(1, round(meta["fps"] * protocol["sampling_interval_s"]))
        tracks, relative, all_boxes = [], {}, []
        for row in self.tracks(video.id)["tracks"]:
            frames, points, boxes = sampled(row["frames"], row["uv"], row["boxes"], step)
            if not len(frames):
                continue
            tracks.append(GridTrack(row["id"], frames, points))
            all_boxes.append(boxes)
            for frame, point, box in zip(frames.tolist(), points, boxes):
                relative[(row["id"], frame)] = box - point[[0, 1, 0, 1]]
        height = vt.median_height(np.concatenate(all_boxes)) if all_boxes else 0.0
        base = Base(tuple(tracks), step, _in_pixels, None, relative, "px", height, "Tracker output")
        windows = history_windows(base.tracks, protocol["history_len"], step)
        predictors = (evaluation.cv_predictor(cfg, protocol["horizon_len"]), evaluation.ca_predictor(cfg, protocol["horizon_len"]))
        rows = [vt.track_row(t.ped_id, t.frames, t.positions, base.boxes(t.positions, np.full(len(t.frames), t.ped_id), t.frames))
                for t in base.tracks]
        prepared = Prepared(Loaded(base, windows, Histories(windows), predictors, {}),
                            {p.name: p.predict(windows.observed) for p in predictors}, rows, step / meta["fps"])
        with self._lock:
            self._prepared[key] = prepared
        return prepared

    def forecast(self, video_id: str, request) -> dict:
        """Forecasts with these errors, on one video. Exploratory, computed live and never saved."""
        from trackvision.platform.services import exploratory as ex
        video = self._video(video_id)
        key = (video.id, request)
        with self._lock:
            if key in self._results:
                self._results.move_to_end(key)
                return self._results[key]
        prepared = self._prepare(video.id)
        loaded, cfg = prepared.loaded, self._config()
        seeds = list(cfg.research_choice("seeds"))
        by_seed = {seed: ex.combine(loaded.tracks, loaded.population, request, seed, cfg, loaded.frame_step,
                                    d_max=loaded.base.d_max, unit="px") for seed in seeds}
        overlay_seed = seeds[OVERLAY_SEED_INDEX]
        out = {"scene": video.id, "exploratory": True, "active": list(request.active), "unit": "px", "label": LABEL, "seeds": seeds,
               "seconds_per_step": prepared.seconds_per_step, "horizon_len": cfg.frozen["protocol"]["static"]["horizon_len"],
               "request": {"a_sigma": request.a_sigma, "b_p": request.b_p, "c_p": request.c_p, "d_p": request.d_p,
                           "reconstruct": request.reconstruct},
               "achieved": _achieved(by_seed),
               "forecasts": {"total": len(loaded.windows),
                             "kept_by_seed": {str(s): int(c.corrupted.predicted.sum()) for s, c in by_seed.items()},
                             "lost_by_reason": _lost(by_seed[overlay_seed])},
               "moved": {p.name: _moved(p, prepared.clean[p.name], by_seed) for p in loaded.predictors},
               "grid": {"frame_step": loaded.frame_step, "tracks": prepared.rows},
               "overlay": _overlay(by_seed[overlay_seed], loaded, overlay_seed)}
        with self._lock:
            self._results[key] = out
            while len(self._results) > KEPT_RESULTS:
                self._results.popitem(last=False)
        return out


def _moved(predictor, clean: np.ndarray, by_seed: dict) -> float | None:
    """The mean distance between each kept forecast with and without errors, over seeds. Change, not accuracy."""
    distances = []
    for combined in by_seed.values():
        corrupted = combined.corrupted
        if corrupted.predicted.any():
            distances.append(np.linalg.norm(predictor.predict(corrupted.inputs) - clean[corrupted.predicted], axis=2).mean())
    return float(np.mean(distances)) if distances else None
