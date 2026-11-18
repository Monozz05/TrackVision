"""Build the browser copies for the Test Videos section (decision 0017).

Each original in `videos/` is first checked against the SHA-1 Wikimedia Commons publishes, and is only ever read. Its copy is
at most 720 pixels high with every frame kept and an even frame rate, so frame k of the copy is frame k of the original at
exactly k / fps seconds, which is what lets the overlay stay locked to the picture. Copies go to the Git-ignored
`data/cache/test_videos/`, apart from every dataset's, and a finished copy is skipped on a rerun.

Run with the project's Python, then run yolo_detect_test_videos.py with the YOLO environment's:

    .venv\\Scripts\\python.exe scripts\\prepare_test_videos.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from trackvision.platform.services import video_cache as vc
from trackvision.platform.services.test_videos import STREET_VIDEOS, StreetVideo

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_HEIGHT = 720


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> tuple[str, int]:
    """The stated frame rate as a fraction, and how many frames the file really holds."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=r_frame_rate,nb_read_packets", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    stream = json.loads(out)["streams"][0]
    return stream["r_frame_rate"], int(stream["nb_read_packets"])


def copy_command(source: Path, rate: str, target: Path) -> list[str]:
    """`-r` before `-i` gives every frame an even timestamp and neither drops nor repeats one."""
    return ["ffmpeg", "-v", "error", "-y", "-r", rate, "-i", str(source), "-vf", f"scale=-2:'min({MAX_HEIGHT},ih)'",
            *vc._ENCODE, str(target)]


def prepare(video: StreetVideo, videos_dir: Path, out_dir: Path) -> dict:
    source = videos_dir / video.file
    if sha1(source) != video.sha1:
        raise ValueError(f"{video.file} is not the file Wikimedia Commons published")
    rate, frames = probe(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / f".{video.id}.{os.getpid()}.building.mp4"
    try:
        subprocess.run(copy_command(source, rate, temporary), check=True, capture_output=True)
        meta = vc._probe(temporary)
        if meta["frames"] != frames:
            raise ValueError(f"{video.id}: the copy has {meta['frames']} frames, but the original has {frames}")
        (out_dir / f"{video.id}.json").write_text(json.dumps(meta), encoding="utf-8")
        os.replace(temporary, out_dir / f"{video.id}.mp4")
    finally:
        temporary.unlink(missing_ok=True)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", type=Path, default=REPO_ROOT / "videos")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "cache" / "test_videos")
    args = parser.parse_args(argv)
    for video in STREET_VIDEOS:
        if (args.out / f"{video.id}.mp4").exists() and (args.out / f"{video.id}.json").exists():
            print(f"{video.id}: already done, skipped")
            continue
        print(f"{video.id}: {prepare(video, args.videos, args.out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
