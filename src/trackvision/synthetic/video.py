"""Lossless video export of rendered synthetic scenes, through the external ffmpeg/ffprobe programs.

System dependency: ffmpeg and ffprobe on PATH (invoked as separate processes, never linked). They are
not Python packages and are not declared in pyproject.toml.

Format: FFV1 (lossless), 8-bit `gray`, Matroska (.mkv), bit-exact flags, metadata stripped,
timestamps passed through. Frame mapping: video frame k = U1 ground-truth frame k (k = 0 .. n-1), at
k / fps seconds; every frame is written, including frames without pedestrians. Reproducibility is
defined on the decoded frame contents (SHA-256 per frame), not on container bytes, which may differ
between ffmpeg versions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from trackvision.synthetic.render import RenderConfig, SceneRenderer
from trackvision.synthetic.scene import SceneGroundTruth

FRAME_MAPPING = "video frame k = ground-truth frame k (k = 0 .. n_frames-1), presentation time k / fps"


class VideoToolError(RuntimeError):
    """Raised when ffmpeg/ffprobe is missing or fails."""


def require_ffmpeg() -> tuple[str, str]:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise VideoToolError("ffmpeg and ffprobe must be on PATH (system dependency)")
    return ffmpeg, ffprobe


def ffmpeg_version() -> str:
    ffmpeg, _ = require_ffmpeg()
    out = subprocess.run([ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True, check=False)
    return out.stdout.splitlines()[0] if out.stdout else "unknown"


def _check_frame(frame: np.ndarray, width: int, height: int) -> bytes:
    if frame.shape != (height, width) or frame.dtype != np.uint8:
        raise ValueError(f"frames must be ({height}, {width}) uint8, got {frame.shape} {frame.dtype}")
    return np.ascontiguousarray(frame).tobytes()


def write_ffv1(path: Path, frames: Iterable[np.ndarray], width: int, height: int, fps: int) -> dict:
    """Stream grey frames into a new FFV1/Matroska file; returns the count and per-frame SHA-256 digests."""
    ffmpeg, _ = require_ffmpeg()
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "gray", "-video_size", f"{width}x{height}", "-framerate", str(fps),
           "-i", "pipe:0",
           "-map", "0:v", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray", "-fps_mode", "passthrough",
           "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact", "-n", str(path)]
    digests, overall = [], hashlib.sha256()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    ok = False
    try:
        for frame in frames:
            data = _check_frame(frame, width, height)
            digests.append(hashlib.sha256(data).digest())
            overall.update(data)
            proc.stdin.write(data)
        ok = True
    except BrokenPipeError:
        pass
    finally:
        try:
            proc.stdin.close()          # always end the input first, or ffmpeg waits for more frames
        except BrokenPipeError:
            pass
        if not ok:
            proc.kill()
        err = proc.stderr.read().decode("utf-8", "replace")
        code = proc.wait()
        if not ok or code != 0:
            path.unlink(missing_ok=True)    # never leave a partial video behind
    if code != 0:
        raise VideoToolError(f"ffmpeg failed ({code}): {err.strip()}")
    return {"n_frames": len(digests), "frame_sha256": digests, "raw_sha256": overall.hexdigest()}


def probe(path: Path) -> dict:
    """Stream facts from ffprobe, counting frames by decoding them."""
    _, ffprobe = require_ffmpeg()
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames",
                          "-of", "json", str(path)], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise VideoToolError(f"ffprobe failed: {out.stderr.strip()}")
    stream = json.loads(out.stdout)["streams"][0]
    stream["nb_read_frames"] = int(stream["nb_read_frames"])
    return stream


def iter_decoded_frames(path: Path, width: int, height: int) -> Iterator[np.ndarray]:
    """Decode a video to (height, width) uint8 grey frames, in order."""
    ffmpeg, _ = require_ffmpeg()
    proc = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
                             "-f", "rawvideo", "-pix_fmt", "gray", "-fps_mode", "passthrough", "pipe:1"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    size = width * height
    completed = False
    try:
        while True:
            chunk = proc.stdout.read(size)
            if not chunk:
                break
            if len(chunk) != size:
                raise VideoToolError("decoded stream ended inside a frame")
            yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width)
        completed = True
    finally:
        if not completed:
            proc.kill()                 # consumer stopped early or decoding failed
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        code = proc.wait()
    if code != 0:
        raise VideoToolError(f"ffmpeg decode failed ({code}): {err.strip()}")


def write_preview_png(path: Path, image: np.ndarray, boxes_xyxy=()) -> Path:
    """A grey PNG of `image` with box outlines (white, with a black inner line), for visual inspection."""
    ffmpeg, _ = require_ffmpeg()
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists")
    canvas = image.copy()
    height, width = canvas.shape
    for x1, y1, x2, y2 in np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4):
        for value, inset in ((255, 0), (0, 1)):
            l, t = int(np.clip(np.floor(x1) + inset, 0, width - 1)), int(np.clip(np.floor(y1) + inset, 0, height - 1))
            r, b = int(np.clip(np.ceil(x2) - 1 - inset, 0, width - 1)), int(np.clip(np.ceil(y2) - 1 - inset, 0, height - 1))
            canvas[t, l:r + 1] = value
            canvas[b, l:r + 1] = value
            canvas[t:b + 1, l] = value
            canvas[t:b + 1, r] = value
    out = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                          "-video_size", f"{width}x{height}", "-i", "pipe:0", "-frames:v", "1", "-c:v", "png",
                          "-n", str(path)], input=canvas.tobytes(), capture_output=True, check=False)
    if out.returncode != 0:
        raise VideoToolError(f"ffmpeg PNG export failed: {out.stderr.decode('utf-8', 'replace').strip()}")
    return path


def export_rendered_scene(scene: SceneGroundTruth, render_config: RenderConfig, out_dir: Path) -> dict:
    """Write video.mkv, ground_truth.npz, render.npz and render_meta.json into a new directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    cam = scene.config.camera
    n_rows = len(scene.frame)
    visible_fraction = np.zeros(n_rows)
    visible_pixels = np.zeros(n_rows, dtype=np.int64)
    silhouette_pixels = np.zeros(n_rows, dtype=np.int64)

    def frames():
        for rendered in SceneRenderer(scene, render_config):
            visible_fraction[rendered.rows] = rendered.visible_fraction
            visible_pixels[rendered.rows] = rendered.visible_pixels
            silhouette_pixels[rendered.rows] = rendered.silhouette_pixels
            yield rendered.image

    written = write_ffv1(out_dir / "video.mkv", frames(), cam.image_width, cam.image_height, scene.config.fps)
    scene.save_npz(out_dir / "ground_truth.npz")
    frame_digests = np.frombuffer(b"".join(written["frame_sha256"]), dtype=np.uint8).reshape(-1, 32)
    np.savez(out_dir / "render.npz", visible_fraction=visible_fraction, visible_pixels=visible_pixels,
             silhouette_pixels=silhouette_pixels, frame_sha256=frame_digests)
    meta = {
        "note": "feasibility fixture; no research claim",
        "scene_content_hash": scene.content_hash(),
        "render_config": dataclasses.asdict(render_config),
        "width": cam.image_width, "height": cam.image_height, "fps": scene.config.fps,
        "n_frames": written["n_frames"], "codec": "ffv1", "pix_fmt": "gray", "container": "matroska",
        "ffmpeg_version": ffmpeg_version(), "raw_sha256": written["raw_sha256"],
        "frame_mapping": FRAME_MAPPING,
    }
    (out_dir / "render_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"dir": out_dir, "meta": meta, "frame_sha256": written["frame_sha256"]}
