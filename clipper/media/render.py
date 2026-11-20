"""ffmpeg rendering of the final 1080x1920 clip.

Layouts (the bandit picks one per clip and learns which wins):
  * blur    - the full 16:9 frame in the middle, a blurred zoom of it filling the rest
  * stack   - streamer webcam on top, gameplay center-crop below
  * crop    - full-height center crop of the gameplay
  * speaker - ClipsAI's speaker-tracking crop (input is already 9:16)
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .captions import Positions

log = logging.getLogger("editor.render")

POSITIONS = {
    "blur": Positions(captions_y=1420, hook_y=330),
    "stack": Positions(captions_y=640, hook_y=100),    # captions sit on the cam/gameplay seam
    "crop": Positions(captions_y=1300, hook_y=250),
    "speaker": Positions(captions_y=1300, hook_y=250),
}


@dataclass
class Probe:
    width: int
    height: int
    duration: float
    fps: float
    has_audio: bool


def probe(path: Path) -> Probe:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"{path} has no video stream")
    num, _, den = (video.get("avg_frame_rate") or "30/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    return Probe(
        width=int(video["width"]), height=int(video["height"]),
        duration=float(data.get("format", {}).get("duration") or video.get("duration") or 0.0),
        fps=fps or 30.0,
        has_audio=any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
    )


def facecam_from_fraction(frac: list[float], p: Probe) -> tuple[int, int, int, int]:
    x, y, w, h = frac
    return _even(x * p.width), _even(y * p.height), _even(w * p.width), _even(h * p.height)


def detect_facecam(path: Path, p: Probe, samples: int = 10) -> tuple[int, int, int, int] | None:
    """Find a streamer webcam overlay: a face that stays put across the clip, off-center.

    In-game faces move between samples and are discarded. Heuristic; for a
    streamer you clip often, set editor.facecams in the config instead.
    """
    try:
        import cv2
    except ImportError:
        return None
    cascade_file = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_file.exists():
        return None
    cascade = cv2.CascadeClassifier(str(cascade_file))
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(p.duration * p.fps)
    frames = []
    for k in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (k + 0.5) / samples))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                         minSize=(p.height // 20, p.height // 20))
        frames.append([tuple(int(v) for v in f) for f in faces])
    cap.release()
    if not frames:
        return None
    clusters: list[dict] = []
    for faces in frames:
        for (x, y, fw, fh) in faces:
            cx, cy = x + fw / 2, y + fh / 2
            for c in clusters:
                if abs(c["cx"] - cx) < 0.04 * p.width and abs(c["cy"] - cy) < 0.04 * p.height \
                        and 0.7 < fw / c["fw"] < 1.4:
                    c["boxes"].append((x, y, fw, fh))
                    break
            else:
                clusters.append({"cx": cx, "cy": cy, "fw": fw, "boxes": [(x, y, fw, fh)]})
    stable = [c for c in clusters if len(c["boxes"]) >= max(3, len(frames) // 2)]
    if not stable:
        return None
    best = max(stable, key=lambda c: len(c["boxes"]))
    boxes = sorted(best["boxes"])
    x, y, fw, fh = boxes[len(boxes) // 2]
    cx, cy = x + fw / 2, y + fh / 2
    if abs(cx - p.width / 2) < 0.15 * p.width and abs(cy - p.height / 2) < 0.15 * p.height:
        return None   # a face in the middle is a full-screen camera, not an overlay
    cam_w, cam_h = min(p.width, fw * 3.2), min(p.height, fh * 2.8)
    x0 = min(max(cx - cam_w / 2, 0), p.width - cam_w)
    y0 = min(max(cy - cam_h * 0.45, 0), p.height - cam_h)
    return _even(x0), _even(y0), _even(cam_w), _even(cam_h)


def _even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


def video_filter(layout: str, subtitles: str, facecam: tuple[int, int, int, int] | None = None) -> str:
    if layout == "crop":
        return f"[0:v]crop=trunc(ih*9/32)*2:ih,scale=1080:1920:flags=lanczos,setsar=1,{subtitles}[v]"
    if layout == "speaker":
        return f"[0:v]scale=1080:1920:flags=lanczos,setsar=1,{subtitles}[v]"
    if layout == "stack":
        if facecam is None:
            raise ValueError("stack layout needs a facecam box")
        x, y, w, h = facecam
        return (
            "[0:v]split=2[cam][game];"
            f"[cam]crop={w}:{h}:{x}:{y},scale=1080:640:force_original_aspect_ratio=increase:flags=lanczos,"
            "crop=1080:640[top];"
            "[game]crop=trunc(ih*27/64)*2:ih,scale=1080:1280:flags=lanczos[bottom];"
            f"[top][bottom]vstack=inputs=2,setsar=1,{subtitles}[v]"
        )
    # blur
    return (
        "[0:v]split=2[bgsrc][fgsrc];"
        "[bgsrc]scale=-2:480,crop=270:480,boxblur=12:2,scale=1080:1920,eq=brightness=-0.07,setsar=1[bg];"
        "[fgsrc]scale=1080:-2:flags=lanczos[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,{subtitles}[v]"
    )


def render(src: Path, dst: Path, layout: str, ass_file: Path, *, facecam: tuple[int, int, int, int] | None = None,
           fonts_dir: str | None = None, crf: int = 20) -> Path:
    """Compose the 9:16 clip with captions burned in."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed")
    p = probe(src)
    fps = min(p.fps, 60.0)
    # ffmpeg runs inside the captions folder, so the filter can name the file without escaping
    subtitles = f"subtitles={ass_file.name}"
    if fonts_dir:
        subtitles += f":fontsdir={Path(fonts_dir).resolve()}"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src.resolve()),
           "-filter_complex", video_filter(layout, subtitles, facecam), "-map", "[v]"]
    if p.has_audio:
        cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-r", f"{fps:.3f}".rstrip("0").rstrip("."), "-movflags", "+faststart", str(dst.resolve())]
    dst.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(cmd, cwd=str(ass_file.parent), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({layout}): {res.stderr.strip()[-800:]}")
    return dst
