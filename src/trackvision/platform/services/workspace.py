"""The scene workspace: which scenes exist, their video, their tracks, and live exploratory injection (decisions 0008, 0011).

ETH, HOTEL and the synthetic scenes work in metres. UCY has no image-to-world calibration and MOT17 has no world
coordinates, so they work in video pixels and no calibration is inferred. The injectors and predictors work on plain
numbers, so they run unchanged. Spec section 6.1.1 keeps MOT17 out of every research result: the owner allowed these
forecasts for live viewing only (decision 0011). Nothing here writes under `runs/`.

Long recordings are also offered in numbered parts (decision 0012). A part is the same footage and annotations cut to a
stretch of time, so it is not a separate scene in any statistic and its values are never pooled.

`evaluation` imports torch, so the injection code is imported on first use and the API starts quickly.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from trackvision import config
from trackvision.platform.config import Settings
from trackvision.platform.services import scene_sweep as ss, tracker_details, video_cache as vc, video_tracks as vt
from trackvision.platform.services.result_reader import ArtifactError, ArtifactMalformed
from trackvision.platform.services.result_views import ViewUnavailable, frozen_protocol
from trackvision.platform.services.run_registry import RunRegistry

MAX_SIGMA_M = 1.0
MAX_SIGMA_PX = 60.0
NOISE_EXAMPLE_PX = (10.0, 30.0, 60.0)
NOISE_EXAMPLE_NAMES = ("Low", "Medium", "High")
NOISE_EXAMPLE_LABEL = "Live example from the frozen injector, not a saved figure. Exploratory, and never saved to runs/."
NOISE_EXAMPLE_PERCENTILE = 0.75      # a person who visibly walks, without picking the most extreme one
OVERLAY_SEED_INDEX = 0
MAX_EVENTS = 400
EWAP_VIDEO = {"ETH": "ewap/ewap_dataset/seq_eth/seq_eth.avi", "HOTEL": "ewap/ewap_dataset/seq_hotel/seq_hotel.avi"}
# ETH's file says 25 fps, but its frames are 6 apart at the annotated 0.4 s, so it was filmed at 15 fps. Played at 25 fps,
# people walk at 2.5 m/s on screen against the 1.5 m/s in their own annotations (decision 0014).
EWAP_FILMED_FPS = {"ETH": 15}
UCY_VIDEO = {"UNIV": "ucy/crowds/data/students003.avi", "ZARA1": "ucy/crowds/data/crowds_zara01.avi",
             "ZARA2": "ucy/crowds/data/crowds_zara02.avi"}
UCY_ANNOTATIONS = {"UNIV": "ucy/crowds/annotations/students003.vsp", "ZARA1": "ucy/crowds/annotations/crowds_zara01.vsp",
                   "ZARA2": "ucy/crowds/annotations/crowds_zara02.vsp"}
UCY_FRAME_SIZE = (720, 576)
MOT17_SEQUENCES = ("MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13")
MOT17_VARIANT = "FRCNN"              # the image folders of the three detection sets are the same footage
MOT17_ARCHIVE = "mot17/MOT17.zip"
SYNTHETIC_SEEDS = {f"SYN-{seed}": seed for seed in range(501, 511)}     # 501 to 505 are the scenes of the stored runs
SYNTHETIC_FRAMES, SYNTHETIC_PEOPLE = 4500, 30       # as learned_feasibility, which cannot be imported without torch
# First annotated frame, the frame after the last, frames per step, number of parts. MOT17-09 and MOT17-13 stay whole:
# one of their halves would keep only 5 windows.
SPLIT_RECORDINGS = {"ETH": (780, 12382, 6, 4), "HOTEL": (1, 18062, 10, 4), "UNIV": (0, 5405, 10, 4), "ZARA1": (0, 9015, 10, 4),
                    "ZARA2": (10, 10519, 10, 4), "MOT17-02": (0, 600, 12, 2), "MOT17-04": (0, 1050, 12, 2),
                    "MOT17-05": (0, 837, 6, 2), "MOT17-10": (0, 654, 12, 2), "MOT17-11": (0, 900, 12, 2)}
PARTS = {f"{name}-{number}": vt.Part(name, number, start, stop) for name, split in SPLIT_RECORDINGS.items()
         for number, (start, stop) in enumerate(vt.part_ranges(*split), start=1)}
DETECTIONS_SCHEMA = "trackvision-detections/1"
NO_ANNOTATIONS = "The annotations for this scene are not on this machine."
PIXEL_CAUTION = "In video pixels, because this scene has no calibration. Values cannot be compared with metres or between scenes."
MOT17_CAUTION = "For viewing only: spec section 6.1.1 keeps MOT17 out of every research result."
MOVING_CAUTION = "The camera moves, so paths and forecasts include its motion."
PART_CAUTION = "Part of a longer recording, not a separate scene, so its values must not be pooled with other parts."
ANNOTATED_NOTE = "Box size is a display estimate. Not everyone in the video is annotated."
UCY_NOTE = f"{ANNOTATED_NOTE} Positions between annotated points are interpolated."
TRACKER_NOTE = "IDs are real tracker output, so tracker mistakes appear as ID changes. Trails follow the image, not the ground."
YOLO_NOTE = f"Boxes are YOLO11 detections, for display only. A very small or hidden person can still be missed. {TRACKER_NOTE}"
TRUTH_NOTE = "Errors run on MOT17's annotated boxes and IDs, sampled every step and interpolated between samples."


class SceneUnavailable(RuntimeError):
    """The scene exists, but this operation is not possible on it."""


@dataclass(frozen=True)
class Base:
    """A scene's true tracks. They stay unchanged for the life of the process."""
    tracks: tuple
    frame_step: int
    to_pixels: Callable
    box_model: vt.BoxModel | None    # display estimate, for scenes that record only a point per person
    true_boxes: dict | None          # (pedestrian, frame) -> box relative to their foot point
    unit: str                        # "m" or "px"
    d_max: float                     # how close two people must be for an identity switch, in `unit`
    source: str
    note: str | None = None
    caution: str | None = None       # about the numbers on this scene, beyond what its catalogue entry says
    seconds_per_step: float | None = None    # set where the frame rate cannot reach the protocol's interval

    def boxes(self, uv, pedestrians, frames) -> np.ndarray:
        """A box per sample. `pedestrians` says who each sample is, which matters once labels have been switched."""
        if self.box_model is not None:
            return self.box_model.boxes(uv)
        relative = np.array([self.true_boxes[(int(p), int(f))] for p, f in zip(pedestrians, frames)]).reshape(-1, 4)
        return np.asarray(uv)[:, [0, 1, 0, 1]] + relative


@dataclass(frozen=True)
class Loaded:
    """A scene ready for injection. Targets stay unchanged for the life of the process."""
    base: Base
    windows: object
    population: object
    predictors: tuple
    clean_errors: dict

    @property
    def tracks(self) -> tuple:
        return self.base.tracks

    @property
    def frame_step(self) -> int:
        return self.base.frame_step


class Workspace:
    def __init__(self, settings: Settings, registry: RunRegistry):
        self.settings = settings
        self.registry = registry
        self.videos = vc.VideoCache(settings.cache_dir)
        self._bases: dict[str, Base] = {}
        self._loaded: dict[str, Loaded] = {}
        self._tracks: dict[tuple, dict] = {}
        self._models: dict[Path, tuple] = {}
        self._results: OrderedDict[tuple, dict] = OrderedDict()
        self._latest: dict[str, object] = {}
        self._running: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._lock = threading.Lock()
        self.sweeps = ss.SweepQueue(self._sweep_job, self._load)

    # --- the catalogue ---------------------------------------------------------------------------

    def _catalogue(self) -> dict[str, dict]:
        raw = self.settings.raw_dir
        scenes: dict[str, dict] = {}

        def add(scene_id, dataset, source, *, errors=True, evidence, regime="static", unit="m", caution=None):
            name = scene_id.replace("MOT17-", "").replace("SYN-", "")
            scenes[scene_id] = {"id": scene_id, "dataset": dataset, "name": name, "parent": None,
                                "regime": regime, "has_video": source, "errors_available": errors,
                                "errors_reason": None if errors else NO_ANNOTATIONS, "evidence_level": evidence,
                                "predictors": ["CV", "CA"] if errors else [], "unit": unit, "caution": caution}
            for part_id, part in PARTS.items():
                if part.recording == scene_id:
                    scenes[part_id] = {**scenes[scene_id], "id": part_id, "name": f"{name} part {part.number}", "parent": scene_id,
                                       "caution": " ".join(filter(None, (PART_CAUTION, caution)))}

        for name, rel in EWAP_VIDEO.items():
            add(name, "EWAP", (raw / rel).is_file(), evidence="descriptive_local_only")
        for name, rel in UCY_VIDEO.items():
            add(name, "UCY", (raw / rel).is_file(), errors=(raw / UCY_ANNOTATIONS[name]).is_file(), unit="px",
                caution=PIXEL_CAUTION, evidence="descriptive_local_only")
        partition = (frozen_protocol(self.settings.base_config).get("datasets") or {}).get("mot17") or {}
        for name in MOT17_SEQUENCES:
            regime = "static" if name in partition.get("static_sequences", []) else "moving"
            present = (raw / MOT17_ARCHIVE).is_file()
            add(name, "MOT17", present, errors=present, evidence="engineering_only", regime=regime, unit="px",
                caution=" ".join((PIXEL_CAUTION, MOT17_CAUTION, *((MOVING_CAUTION,) if regime == "moving" else ()))))
        for name in SYNTHETIC_SEEDS:
            add(name, "Synthetic", True, evidence="engineering_only")
        return scenes

    def _stored_run(self, entry: dict) -> str | None:
        if entry["parent"]:
            return None
        if entry["dataset"] == "MOT17":
            run = self.registry.newest(("mot17_tracking",))
            return run.id if run else None
        for run in self.registry.runs():
            if run.kind == "sweep" and run.usable and "P-4" in run.summary["experiments"] and entry["id"] in run.summary["scenes"]:
                return run.id
        return None

    def _detector(self, entry: dict) -> str | None:
        """The model that produced this scene's boxes, or None if none have been generated.

        The catalogue asks this for every scene, and these files are large, so a file is read once until it changes.
        """
        recording = entry["parent"] or entry["id"]
        path = self.settings.detections_dir / f"{recording}.json"
        if entry["dataset"] == "Synthetic" or not path.is_file():
            return None
        stamp = path.stat().st_mtime_ns
        with self._lock:
            known = self._models.get(path)
        if known and known[0] == stamp:
            return known[1]
        try:
            model = self._detections(recording).get("model")
        except ArtifactError:
            model = None
        with self._lock:
            self._models[path] = (stamp, model)
        return model

    def _tracked(self, entry: dict) -> bool:
        """Whether a scene shows tracker output by default: MOT17 always, other videos once YOLO has run on them."""
        return entry["dataset"] == "MOT17" or self._detector(entry) is not None

    def _detections(self, scene_id: str) -> dict:
        doc = self.registry.reader.read_json(self.settings.detections_dir / f"{scene_id}.json", cache=False)
        frames = doc.get("by_frame")
        if doc.get("schema") != DETECTIONS_SCHEMA or doc.get("scene") != scene_id or not isinstance(frames, list) \
                or len(frames) != doc.get("frames"):
            raise ArtifactMalformed(f"{scene_id}.json is not a valid detection file")
        return doc

    def controls(self) -> dict:
        grids = (frozen_protocol(self.settings.base_config).get("injection") or {}).get("grids") or {}

        def top(code):
            return max((grids.get(code) or {}).get("levels") or [None], key=lambda v: -1 if v is None else v)

        return {"A": {"max": MAX_SIGMA_M, "unit": "m", "protocol_max": top("A")},
                "B": {"max": 1.0, "unit": "fraction", "protocol_max": top("B")},
                "C": {"max": 1.0, "unit": "fraction", "protocol_max": top("C")},
                "D": {"max": 1.0, "unit": "fraction", "protocol_max": top("D")}}

    def pixel_controls(self) -> dict:
        """No sweep was ever run in pixels, so these sliders have no tested range."""
        return {"A": {"max": MAX_SIGMA_PX, "unit": "px", "protocol_max": None},
                **{code: {"max": 1.0, "unit": "fraction", "protocol_max": None} for code in "BCD"}}

    def noise_limit(self, scene_id: str) -> float:
        return MAX_SIGMA_PX if self.entry(scene_id)["unit"] == "px" else MAX_SIGMA_M

    def scenes(self) -> dict:
        out = []
        for entry in self._catalogue().values():
            out.append({**entry, "stored_run": self._stored_run(entry), "detector": self._detector(entry),
                        "video_state": self.videos.status(entry["id"])["state"] if entry["has_video"] else vc.MISSING})
        return {"items": out, "controls": self.controls(), "pixel_controls": self.pixel_controls(),
                "ffmpeg_available": vc.ffmpeg_available(),
                "exploratory_label": "Exploratory, not a research result."}

    def entry(self, scene_id: str) -> dict:
        entry = self._catalogue().get(scene_id)
        if entry is None:
            raise ViewUnavailable(f"there is no scene called {scene_id}")
        return entry

    # --- video -----------------------------------------------------------------------------------

    def _builder(self, entry: dict) -> Callable[[Path], None]:
        raw, scene_id = self.settings.raw_dir, entry["parent"] or entry["id"]
        part = PARTS.get(entry["id"])
        frames = (part.start, part.stop) if part else None
        if entry["dataset"] == "EWAP":
            return vc.from_file(raw / EWAP_VIDEO[scene_id], frames, EWAP_FILMED_FPS.get(scene_id))
        if entry["dataset"] == "UCY":
            return vc.from_file(raw / UCY_VIDEO[scene_id], frames)
        if entry["dataset"] == "MOT17":
            return self._mot17_builder(scene_id, frames)
        return self._synthetic_builder(scene_id)

    def _mot17_builder(self, sequence: str, frames: tuple[int, int] | None = None) -> Callable[[Path], None]:
        import zipfile

        from trackvision.io import mot17
        archive = self.settings.raw_dir / MOT17_ARCHIVE

        def build(target: Path) -> None:
            with zipfile.ZipFile(archive) as zf:
                info = mot17.read_seqinfo(zf, sequence, MOT17_VARIANT)
                start, stop = frames or (0, info.seq_length)
                members = [mot17.member(zf, sequence, MOT17_VARIANT, f"{info.im_dir}/{k + 1:06d}{info.im_ext}")
                           for k in range(start, stop)]
            vc.from_frames(vc.zip_images(archive, members), info.frame_rate, ["-f", "image2pipe", "-c:v", "mjpeg"],
                           ["-vf", f"scale=-2:{vt.VIDEO_HEIGHT}"])(target)
        return build

    def _synthetic_builder(self, scene_id: str) -> Callable[[Path], None]:
        def build(target: Path) -> None:
            from trackvision.synthetic.render import RenderConfig, SceneRenderer
            from trackvision.synthetic.scene import SceneConfig, generate
            dt = self._config().frozen["protocol"]["static"]["sampling_interval_s"]
            scene = generate(SceneConfig(seed=SYNTHETIC_SEEDS[scene_id], n_frames=SYNTHETIC_FRAMES, n_pedestrians=SYNTHETIC_PEOPLE), dt)
            camera = scene.config.camera

            def frames():
                for rendered in SceneRenderer(scene, RenderConfig()):
                    yield np.asarray(rendered.image, dtype=np.uint8).tobytes()
            vc.from_frames(frames, scene.config.fps, ["-f", "rawvideo", "-pix_fmt", "gray",
                                                      "-s", f"{camera.image_width}x{camera.image_height}"])(target)
        return build

    def video_status(self, scene_id: str) -> dict:
        """Report the state, starting the first build if the video is not ready yet."""
        entry = self.entry(scene_id)
        if not entry["has_video"]:
            return {"state": vc.MISSING, "reason": "The source video for this scene is not on this machine."}
        state = self.videos.status(scene_id)
        if state["state"] == vc.READY:
            return state
        if not vc.ffmpeg_available():
            return {"state": vc.FAILED, "reason": "ffmpeg and ffprobe must be on PATH to prepare video."}
        return self.videos.request(scene_id, self._builder(entry))

    def video_file(self, scene_id: str) -> Path:
        self.entry(scene_id)
        if self.videos.status(scene_id)["state"] != vc.READY:
            raise ViewUnavailable(f"the video for {scene_id} is not ready")
        return self.videos.path(scene_id)

    # --- tracks and injection --------------------------------------------------------------------

    def _config(self):
        return config.validate().config

    def _base(self, scene_id: str) -> Base:
        """A scene's true tracks. This needs neither torch nor the predictors, so tracks open quickly."""
        with self._lock:
            if scene_id in self._bases:
                return self._bases[scene_id]
        cfg = self._config()
        dataset = self.entry(scene_id)["dataset"]
        d_max = float(cfg.frozen["injection"]["d_max_m"])
        if scene_id in PARTS:
            base = self._part_base(PARTS[scene_id])
        elif dataset == "EWAP":
            from trackvision.io import ewap
            scene = ewap.load_scene(scene_id, cfg, raw_root=self.settings.raw_dir / "ewap")
            base = Base(tuple(scene.tracks), scene.frame_step, scene.transform.world_to_image, vt.SCENE_BOXES[scene_id], None,
                        "m", d_max, "Human annotations", ANNOTATED_NOTE)
        elif dataset == "UCY":
            base = self._ucy_base(scene_id)
        elif dataset == "MOT17":
            base = self._mot17_base(scene_id, cfg)
        else:
            from trackvision.synthetic.scene import SceneConfig, generate
            from trackvision.synthetic.tracks import ground_truth_grid_tracks
            dt = cfg.frozen["protocol"]["static"]["sampling_interval_s"]
            truth = generate(SceneConfig(seed=SYNTHETIC_SEEDS[scene_id], n_frames=SYNTHETIC_FRAMES, n_pedestrians=SYNTHETIC_PEOPLE), dt)
            relative = truth.bbox_xyxy - truth.foot_image[:, [0, 1, 0, 1]]
            true_boxes = {(int(p), int(f)): box for p, f, box in zip(truth.ped_id, truth.frame, relative)}
            base = Base(ground_truth_grid_tracks(truth), truth.frames_per_step, truth.config.camera.ground_transform().world_to_image,
                        None, true_boxes, "m", d_max, "Scene generator")
        with self._lock:
            self._bases[scene_id] = base
        return base

    def _part_base(self, part: vt.Part) -> Base:
        """The recording's tracks and settings, cut to the part. Its unit, boxes and switch distance stay the recording's."""
        whole = self._base(part.recording)
        true_boxes = whole.true_boxes and {(person, frame - part.start): box for (person, frame), box in whole.true_boxes.items()
                                           if part.start <= frame < part.stop}
        return replace(whole, tracks=vt.cut_tracks(whole.tracks, part), true_boxes=true_boxes, note=_part_note(part, whole.note))

    def _ucy_base(self, scene_id: str) -> Base:
        path = self.settings.raw_dir / UCY_ANNOTATIONS[scene_id]
        if not path.is_file():
            raise ViewUnavailable(f"the annotations for {scene_id} are not on this machine")
        model = vt.SCENE_BOXES[scene_id]
        tracks = vt.ucy_tracks(path, *UCY_FRAME_SIZE)
        height = vt.median_height(model.boxes(np.concatenate([t.positions for t in tracks])))
        return Base(tracks, vt.UCY_FRAME_STEP, _in_pixels, model, None, "px", height, "Human annotations", UCY_NOTE)

    def _mot17_base(self, scene_id: str, cfg) -> Base:
        archive = self.settings.raw_dir / MOT17_ARCHIVE
        if not archive.is_file():
            raise ViewUnavailable(f"the ground truth for {scene_id} is not on this machine")
        interval = cfg.frozen["protocol"]["static"]["sampling_interval_s"]
        try:
            truth = vt.mot17_truth(archive, scene_id, interval)
        except KeyError as error:
            raise ViewUnavailable(f"the ground truth for {scene_id} is not on this machine") from error
        off_grid = abs(truth.seconds_per_step - interval) > 1e-9
        step = f"One step here is {truth.seconds_per_step:.2f} s, not {interval} s." if off_grid else None
        return Base(truth.tracks, truth.frame_step, _in_pixels, None, truth.relative_boxes, "px", truth.person_height,
                    "MOT17 ground truth", " ".join(filter(None, (TRUTH_NOTE, step))), step, truth.seconds_per_step)

    def _load(self, scene_id: str) -> Loaded:
        entry = self.entry(scene_id)
        if not entry["errors_available"]:
            raise SceneUnavailable(entry["errors_reason"])
        with self._lock:
            if scene_id in self._loaded:
                return self._loaded[scene_id]
        from trackvision.experiments import evaluation
        from trackvision.experiments.learned_feasibility import SyntheticScene
        from trackvision.predict.windowing import static_windows
        cfg = self._config()
        base = self._base(scene_id)
        scene = SyntheticScene(scene_id, base.tracks, base.frame_step, {})
        windows = static_windows(scene, cfg)
        windows.target.setflags(write=False)
        horizon = windows.horizon_len
        predictors = (evaluation.cv_predictor(cfg, horizon), evaluation.ca_predictor(cfg, horizon))
        loaded = Loaded(base, windows, evaluation.scene_population(base.tracks, windows), predictors,
                        {p.name: evaluation.clean_errors(p, windows) for p in predictors})
        with self._lock:
            self._loaded[scene_id] = loaded
        return loaded

    def _detector_label(self, scene_id: str) -> tuple[str, dict | None]:
        try:
            document = self._detections(scene_id)
        except ArtifactError:
            return "MOT17 public FRCNN detections", None
        return f"{document.get('model')} detections", document

    def clean_tracks(self, scene_id: str, source: str = "default") -> dict:
        """Every person in a scene, with an ID, a position and a box, for any dataset.

        A scene with a tracker shows it by default. `truth` asks for the annotated people, which is what errors are
        injected into.
        """
        entry = self.entry(scene_id)
        dataset = entry["dataset"]
        tracker = source == "default" and self._tracked(entry)
        part = PARTS.get(scene_id)
        if tracker and part:
            whole = self.clean_tracks(part.recording)
            return {**whole, "scene": scene_id, "note": _part_note(part, whole["note"]), "tracks": vt.cut_rows(whole["tracks"], part)}
        if tracker:
            detector, document = self._detector_label(scene_id)
            key = (scene_id, *((document.get("weights_sha256"), document.get("imgsz")) if document else (None,)))
        else:
            detector, document, key = None, None, (scene_id, "truth")
        with self._lock:
            if key in self._tracks:
                return self._tracks[key]
        static = (frozen_protocol(self.settings.base_config).get("protocol") or {}).get("static") or {}
        out = {"scene": scene_id, "frame_step": 1 if tracker else None, "history_len": static.get("history_len"),
               "horizon_len": static.get("horizon_len"), "note": None}
        if tracker and dataset != "MOT17":
            out.update(source=f"IoU tracker on {detector}", note=YOLO_NOTE, tracks=vt.video_tracks(document))
        elif tracker:
            archive = self.settings.raw_dir / MOT17_ARCHIVE
            if not archive.is_file():
                raise ViewUnavailable(f"the tracking input for {scene_id} is not on this machine")
            out.update(source=f"IoU tracker on {detector}", note=YOLO_NOTE if document else TRACKER_NOTE,
                       tracks=vt.mot17_tracks(archive, scene_id, document))
        else:
            base = self._base(scene_id)
            rows = []
            for t in base.tracks:
                uv = base.to_pixels(np.asarray(t.positions, dtype=np.float64))
                rows.append(vt.track_row(t.ped_id, t.frames, uv, base.boxes(uv, np.full(len(t.frames), t.ped_id), t.frames)))
            out.update(source=base.source, frame_step=base.frame_step, note=base.note, tracks=rows)
        with self._lock:
            self._tracks[key] = out
        return out

    def inject(self, scene_id: str, request) -> dict:
        """One request at a time per scene. A request the user has moved on from is dropped, so moving a slider several
        times on a large scene never queues up several slow calculations."""
        key = (scene_id, request)
        with self._lock:
            if key in self._results:
                self._results.move_to_end(key)
                return self._results[key]
            self._latest[scene_id] = request
            running = self._running[scene_id]
        with running:
            with self._lock:
                if key in self._results:
                    return self._results[key]
            result = self._compute(scene_id, request)
            with self._lock:
                self._results[key] = result
                while len(self._results) > 12:
                    self._results.popitem(last=False)
        return result

    def noise_example(self, scene_id: str) -> dict:
        """One real person's clean history, that history at three noise levels, and their true future (decision 0013).

        Metre scenes use levels from the frozen grid. No grid exists in pixels, so those levels are the slider's range.
        """
        from trackvision.platform.services import exploratory as ex
        loaded = self._load(scene_id)
        base, cfg, windows = loaded.base, self._config(), loaded.windows
        grid = cfg.frozen["injection"]["grids"]["A"]["levels"]
        sigmas = NOISE_EXAMPLE_PX if base.unit == "px" else (grid[1], grid[3], grid[5])
        seed = list(cfg.research_choice("seeds"))[OVERLAY_SEED_INDEX]
        index = _example_window(windows)
        levels = []
        for name, sigma in zip(NOISE_EXAMPLE_NAMES, sigmas):
            combined = ex.combine(loaded.tracks, loaded.population, ex.ErrorRequest(a_sigma=float(sigma)), seed, cfg,
                                  loaded.frame_step, d_max=base.d_max, unit=base.unit)
            levels.append({"name": name, "sigma": float(sigma), "noisy": combined.corrupted.inputs[index].tolist()})
        return {"scene": scene_id, "exploratory": True, "unit": base.unit, "label": NOISE_EXAMPLE_LABEL, "seed": seed,
                "person": int(windows.track_ids[index]), "frames": windows.frames[index].tolist(),
                "image_coordinates": base.unit == "px", "observed": windows.observed[index].tolist(),
                "future": windows.target[index].tolist(), "levels": levels}

    def _sweep_job(self, scene_id: str) -> ss.Job:
        cfg = self._config()
        levels = ss.tested_levels(cfg, self.entry(scene_id)["unit"], MAX_SIGMA_PX)
        return ss.Job(scene_id, levels, list(cfg.research_choice("seeds")), cfg, cfg.frozen["analysis"]["inverse_requirement_q"])

    def sweep(self, scene_id: str) -> dict:
        """Every error at every tested level, as far as it has got. It runs in the background and is never stored."""
        entry = self.entry(scene_id)
        if not entry["errors_available"]:
            raise SceneUnavailable(entry["errors_reason"])
        return self.sweeps.request(scene_id, entry)

    def tracker(self, scene_id: str) -> dict:
        entry = self.entry(scene_id)
        overlay = None
        if entry["dataset"] == "MOT17":
            detector = self._detector_label(entry["parent"] or scene_id)[0]
            overlay = f"IoU tracker on {detector}, score {vt.MIN_DETECTION_SCORE} and above, for display"
        return tracker_details.details(entry, self.registry, self.settings.base_config, overlay)

    def _require_latest(self, scene_id: str, request) -> None:
        with self._lock:
            if self._latest.get(scene_id) != request:
                raise SceneUnavailable("a newer request replaced this one")

    def _compute(self, scene_id: str, request) -> dict:
        from trackvision.platform.services import exploratory as ex
        self._require_latest(scene_id, request)
        loaded = self._load(scene_id)
        base, cfg = loaded.base, self._config()
        seeds = list(cfg.research_choice("seeds"))
        by_seed = {}
        for seed in seeds:
            self._require_latest(scene_id, request)
            by_seed[seed] = ex.combine(loaded.tracks, loaded.population, request, seed, cfg, loaded.frame_step,
                                       d_max=base.d_max, unit=base.unit)
        label = ex.LABEL if len(request.active) > 1 else ex.SINGLE_LABEL
        return {
            "scene": scene_id, "exploratory": True, "active": list(request.active), "unit": base.unit,
            "label": " ".join(filter(None, (label, self.entry(scene_id)["caution"], base.caution))), "seeds": seeds,
            "request": {"a_sigma": request.a_sigma, "b_p": request.b_p, "c_p": request.c_p, "d_p": request.d_p,
                        "reconstruct": request.reconstruct},
            "achieved": _achieved(by_seed),
            "predictors": ex.summarise(loaded.windows, loaded.predictors, loaded.clean_errors, by_seed),
            "windows": {"eligible": len(loaded.windows), "predicted_by_seed": {
                str(s): int(c.corrupted.predicted.sum()) for s, c in by_seed.items()},
                "lost_by_reason": _lost(by_seed[seeds[OVERLAY_SEED_INDEX]])},
            "overlay": _overlay(by_seed[seeds[OVERLAY_SEED_INDEX]], loaded, seeds[OVERLAY_SEED_INDEX]),
        }


def _in_pixels(uv) -> np.ndarray:
    """UCY and MOT17 tracks are already in video pixels."""
    return np.asarray(uv, dtype=np.float64)


def _example_window(windows) -> int:
    """The window at a fixed percentile of distance walked, so the choice is repeatable."""
    points = np.concatenate([windows.observed, windows.target], axis=1)
    walked = np.linalg.norm(np.diff(points, axis=1), axis=2).sum(axis=1)
    return int(np.argsort(walked, kind="stable")[int(NOISE_EXAMPLE_PERCENTILE * (len(walked) - 1))])


def _part_note(part: vt.Part, note: str | None) -> str:
    return " ".join(filter(None, (f"Part {part.number} of {part.recording}. IDs are those of the full recording.", note)))


def _lost(combined) -> dict:
    counts: dict[str, int] = {}
    for reason in combined.corrupted.reason:
        if reason is not None:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _achieved(by_seed: dict) -> dict:
    """Each error's achieved severity by seed and the mean over seeds, from the injectors' own reports."""
    out: dict = {}
    for seed, combined in by_seed.items():
        for code, report in combined.achieved.items():
            row = out.setdefault(code, {"measure": report["measure"], "by_seed": {}})
            row["by_seed"][str(seed)] = report["value"]
    for row in out.values():
        values = [v for v in row["by_seed"].values() if v is not None]
        row["mean"] = float(np.mean(values)) if values else None
    return out


def _forecasts(combined, loaded: Loaded, predictor) -> dict:
    """Each predicted window's forecast in pixels, grouped by the last frame of its history."""
    corrupted, win, out = combined.corrupted, loaded.windows, {}
    if corrupted.predicted.any():
        index = np.flatnonzero(corrupted.predicted)
        last = win.frames[index, win.history_len - 1]
        for frame, ped, path in zip(last.tolist(), win.track_ids[index].tolist(), predictor.predict(corrupted.inputs)):
            uv = loaded.base.to_pixels(np.asarray(path, dtype=np.float64))
            out.setdefault(str(frame), []).append([ped, np.round(uv, 1).tolist()])
    return out


def _overlay(combined, loaded: Loaded, seed: int) -> dict:
    """What one seed did to the tracks, in pixels, for drawing on the video."""
    tracks = []
    for t in combined.labelled:
        label = int(t.ped_id)
        positions = t.positions + combined.offsets[label] if label in combined.offsets else t.positions
        uv = loaded.base.to_pixels(np.asarray(positions, dtype=np.float64))
        people = combined.occupant[label] if combined.occupant else np.full(len(t.frames), label)
        row = vt.track_row(label, t.frames, uv, loaded.base.boxes(uv, people, t.frames))
        if label in combined.present:
            row["dropped"] = np.flatnonzero(~combined.present[label]).tolist()
        if label in combined.segment:
            row["segment"] = combined.segment[label].tolist()
        tracks.append(row)
    frames = {int(t.ped_id): t.frames for t in combined.labelled}
    events = [{"frame": int(e.frame), "kind": "switch", "a": int(e.initiator), "b": int(e.partner)}
              for e in combined.switch_events]
    splits = [{"frame": int(frames[label][k]), "kind": "split", "a": label, "b": None} for label, k in combined.split_events]
    splits.sort(key=lambda e: e["frame"])
    # Identity switches are always kept, so a long run of breaks can never push one off the timeline.
    events = sorted(events + splits[:max(0, MAX_EVENTS - len(events))], key=lambda e: e["frame"])
    return {"seed": seed, "tracks": tracks, "events": events, "events_total": len(combined.switch_events) + len(splits),
            "predicted": {p.name: _forecasts(combined, loaded, p) for p in loaded.predictors}}
