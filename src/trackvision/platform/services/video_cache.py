"""Browser-playable copies of scene video, built once with ffmpeg and kept in the Git-ignored `data/cache/`.

Browsers cannot play the datasets' AVI files, MOT17 stores images inside its ZIP, and synthetic scenes are rendered.
So a scene's first playback builds an H.264 copy in the background. Source files are only read: `data/raw` is never
written and the MOT17 archive is never extracted. Every frame is kept in order, so video frame k stays source frame k.
A part of a long recording keeps an exact stretch of frames, so its frame k is source frame start + k. A file whose header
states the wrong rate can be given the right one, which changes only when each frame is shown (decision 0014).
These are viewing copies for this machine, which is why they live outside `runs/` and are never committed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

log = logging.getLogger("trackvision.platform")
READY, PREPARING, MISSING, FAILED = "ready", "preparing", "missing", "failed"
_ENCODE = ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-g", "25", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-fps_mode", "passthrough"]


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def from_file(source: Path, frames: tuple[int, int] | None = None, fps: float | None = None) -> Callable[[Path], None]:
    """Build from a video file on disk, or from only its frames [start, stop), which then count from 0.

    `fps` replaces a wrong rate in the file's header. Every frame is kept, and only their timing changes.
    """
    cut = ["-vf", f"trim=start_frame={frames[0]}:end_frame={frames[1]},setpts=PTS-STARTPTS"] if frames else []
    rate = ["-r", str(fps)] if fps else []

    def build(target: Path) -> None:
        subprocess.run(["ffmpeg", "-v", "error", "-y", *rate, "-i", str(source), *cut, *_ENCODE, str(target)], check=True,
                       capture_output=True)
    return build


def from_frames(frames: Callable[[], Iterable[bytes]], fps: float, input_args: list[str],
                filters: list[str] | None = None) -> Callable[[Path], None]:
    """Build from encoded or raw frames piped to ffmpeg, so that nothing is written outside the cache."""
    def build(target: Path) -> None:
        process = subprocess.Popen(["ffmpeg", "-v", "error", "-y", *input_args, "-framerate", str(fps), "-i", "-",
                                    *(filters or []), *_ENCODE, str(target)], stdin=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)
        try:
            for frame in frames():
                process.stdin.write(frame)
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, "ffmpeg")
    return build


def zip_images(archive: Path, members: list[str]) -> Callable[[], Iterable[bytes]]:
    """Read image members from a ZIP in memory. The archive is opened read-only and nothing is extracted."""
    def frames() -> Iterable[bytes]:
        with zipfile.ZipFile(archive) as zf:
            for name in members:
                yield zf.read(name)
    return frames


class VideoCache:
    def __init__(self, cache_dir: Path, max_workers: int = 2):
        self.cache_dir = Path(cache_dir)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="video")
        self._jobs: dict[str, object] = {}
        self._failed: set[str] = set()
        self._lock = threading.Lock()

    def path(self, scene_id: str) -> Path:
        return self.cache_dir / f"{scene_id}.mp4"

    def _meta_path(self, scene_id: str) -> Path:
        return self.cache_dir / f"{scene_id}.json"

    def meta(self, scene_id: str) -> dict | None:
        try:
            return json.loads(self._meta_path(scene_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def status(self, scene_id: str) -> dict:
        """The current state, without starting anything."""
        if self.path(scene_id).is_file() and self.meta(scene_id):
            return {"state": READY, **self.meta(scene_id)}
        with self._lock:
            job = self._jobs.get(scene_id)
            if job is not None and not job.done():
                return {"state": PREPARING}
            return {"state": FAILED if scene_id in self._failed else MISSING}

    def request(self, scene_id: str, build: Callable[[Path], None]) -> dict:
        """Start building if needed, and report the state. Only one build runs per scene."""
        state = self.status(scene_id)
        if state["state"] in (READY, PREPARING):
            return state
        with self._lock:
            job = self._jobs.get(scene_id)
            if job is None or job.done():
                self._failed.discard(scene_id)
                self._jobs[scene_id] = self._pool.submit(self._build, scene_id, build)
        return {"state": PREPARING}

    def _build(self, scene_id: str, build: Callable[[Path], None]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_dir / f".{scene_id}.{os.getpid()}.building.mp4"
        try:
            build(temporary)
            meta = _probe(temporary)
            self._meta_path(scene_id).write_text(json.dumps(meta), encoding="utf-8")
            os.replace(temporary, self.path(scene_id))
        except Exception:
            log.exception("could not prepare video for %s", scene_id)
            with self._lock:
                self._failed.add(scene_id)
        finally:
            temporary.unlink(missing_ok=True)


def _probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    stream = json.loads(out)["streams"][0]
    top, bottom = stream["r_frame_rate"].split("/")
    return {"width": int(stream["width"]), "height": int(stream["height"]), "fps": float(top) / float(bottom),
            "frames": int(stream["nb_read_packets"])}
