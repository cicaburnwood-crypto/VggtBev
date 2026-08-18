#!/usr/bin/env python3
"""Forward/reverse RGB-sequence playback with live RGB-only P1B inference."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import posixpath
import re
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image


FRONT_PLAYLIST_SUFFIX = "__uid_s_1000__uid_e_video.m3u8"
SOURCE_SAMPLE_HZ = 2.0
MAX_HISTORY = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--model-server-url", default="http://127.0.0.1:8898")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8897)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def _duration_label(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, remainder = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remainder:02d}"
    return f"{minutes:d}:{remainder:02d}"


@dataclass(frozen=True)
class VideoTrack:
    archive_name: str
    playlist_member: str
    label: str
    duration_seconds: float
    rotation_degrees: int
    segment_members: tuple[str, ...]
    source_kind: str = "hls_mpegts"

    @property
    def cache_key(self) -> str:
        digest = hashlib.sha256(
            f"{self.archive_name}\0{self.playlist_member}".encode("utf-8")
        ).hexdigest()
        return digest[:24]

    def public(self) -> dict[str, object]:
        return {
            "id": self.playlist_member,
            "label": self.label,
            "duration_seconds": self.duration_seconds,
            "duration_label": _duration_label(self.duration_seconds),
        }


class ArchiveCatalog:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        if self.data_dir.is_file():
            if self.data_dir.suffix.lower() != ".mp4":
                raise ValueError(f"unsupported video file: {self.data_dir}")
            self.source_root = self.data_dir.parent
            self.archives = (self.data_dir,)
        elif self.data_dir.is_dir():
            self.source_root = self.data_dir
            self.archives = tuple(
                sorted(
                    {
                        *self.data_dir.rglob("minirover-*.tar"),
                        *self.data_dir.rglob("*.tgz"),
                        *self.data_dir.rglob("*.mp4"),
                    }
                )
            )
        else:
            raise FileNotFoundError(f"video source does not exist: {self.data_dir}")
        if not self.archives:
            raise FileNotFoundError(
                f"no supported TAR, TGZ, or MP4 video sources in {self.data_dir}"
            )
        self.by_name = {
            path.relative_to(self.source_root).as_posix(): path
            for path in self.archives
        }
        self.scan_cache: dict[str, tuple[VideoTrack, ...]] = {}
        self.lock = threading.Lock()

    def archive_list(self) -> list[dict[str, str]]:
        return [
            {
                "id": path.relative_to(self.source_root).as_posix(),
                "label": (
                    path.name.removesuffix(".tgz")
                    .removesuffix(".tar")
                    .removesuffix(".mp4")
                ),
            }
            for path in self.archives
        ]

    def archive_path(self, archive_name: str) -> Path:
        try:
            return self.by_name[archive_name]
        except KeyError as error:
            raise ValueError("unknown archive") from error

    def tracks(self, archive_name: str) -> tuple[VideoTrack, ...]:
        self.archive_path(archive_name)
        with self.lock:
            cached = self.scan_cache.get(archive_name)
        if cached is not None:
            return cached

        archive_path = self.archive_path(archive_name)
        if archive_path.suffix.lower() == ".mp4":
            tracks = self._direct_video_tracks(archive_name, archive_path)
        else:
            with tarfile.open(archive_path, mode="r:*") as archive:
                members = {
                    member.name: member
                    for member in archive.getmembers()
                    if member.isfile()
                }
                tracks = self._hls_tracks(archive_name, archive, members)
                if not tracks:
                    tracks = self._tum_rgb_tracks(archive_name, archive, members)
        result = tuple(tracks)
        with self.lock:
            self.scan_cache[archive_name] = result
        return result

    @staticmethod
    def _direct_video_tracks(
        archive_name: str,
        video_path: Path,
    ) -> list[VideoTrack]:
        command = [
            "/usr/bin/ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        duration = float(result.stdout.strip())
        label = f"{video_path.stem} · {_duration_label(duration)}"
        return [
            VideoTrack(
                archive_name=archive_name,
                playlist_member=archive_name,
                label=label,
                duration_seconds=duration,
                rotation_degrees=0,
                segment_members=(),
                source_kind="direct_video",
            )
        ]

    @staticmethod
    def _hls_tracks(
        archive_name: str,
        archive: tarfile.TarFile,
        members: dict[str, tarfile.TarInfo],
    ) -> list[VideoTrack]:
        tracks: list[VideoTrack] = []
        playlist_names = sorted(
            name for name in members if name.endswith(FRONT_PLAYLIST_SUFFIX)
        )
        for playlist_name in playlist_names:
            source = archive.extractfile(members[playlist_name])
            if source is None:
                continue
            text = source.read().decode("utf-8", errors="replace")
            duration = 0.0
            segment_members: list[str] = []
            parent = posixpath.dirname(playlist_name)
            rotation = 0
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if line.startswith("#EXTINF:"):
                    value = line.removeprefix("#EXTINF:").split(",", 1)[0]
                    try:
                        duration += float(value)
                    except ValueError:
                        pass
                elif line.startswith("#EXT-X-ROTATE:"):
                    match = re.search(r"ROTATE=(-?\d+)", line)
                    if match:
                        rotation = int(match.group(1)) % 360
                elif line and not line.startswith("#"):
                    clean_name = line.split("?", 1)[0]
                    member_name = posixpath.normpath(
                        posixpath.join(parent, clean_name)
                    )
                    if not member_name.startswith(parent + "/"):
                        raise ValueError("playlist segment escapes its ride directory")
                    if member_name not in members:
                        raise ValueError(
                            f"playlist segment is absent from TAR: {member_name}"
                        )
                    segment_members.append(member_name)
            if not segment_members:
                continue
            ride_name = playlist_name.split("/", 1)[0]
            tracks.append(
                VideoTrack(
                    archive_name=archive_name,
                    playlist_member=playlist_name,
                    label=f"{ride_name} · {_duration_label(duration)}",
                    duration_seconds=duration,
                    rotation_degrees=rotation,
                    segment_members=tuple(segment_members),
                    source_kind="hls_mpegts",
                )
            )
        return tracks

    @staticmethod
    def _tum_rgb_tracks(
        archive_name: str,
        archive: tarfile.TarFile,
        members: dict[str, tarfile.TarInfo],
    ) -> list[VideoTrack]:
        tracks: list[VideoTrack] = []
        index_names = sorted(name for name in members if name.endswith("/rgb.txt"))
        for index_name in index_names:
            source = archive.extractfile(members[index_name])
            if source is None:
                continue
            parent = posixpath.dirname(index_name)
            timestamped_members: list[tuple[float, str]] = []
            for raw_line in source.read().decode("utf-8", errors="replace").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 2:
                    continue
                try:
                    timestamp = float(fields[0])
                except ValueError:
                    continue
                member_name = posixpath.normpath(
                    posixpath.join(parent, fields[1])
                )
                if not member_name.startswith(parent + "/"):
                    raise ValueError("TUM RGB frame escapes its sequence directory")
                if member_name not in members:
                    raise ValueError(f"TUM RGB frame is absent from TGZ: {member_name}")
                timestamped_members.append((timestamp, member_name))
            timestamped_members.sort()
            if not timestamped_members:
                continue
            start_time = timestamped_members[0][0]
            next_sample_time = start_time
            sample_period = 1.0 / SOURCE_SAMPLE_HZ
            sampled_members: list[str] = []
            for timestamp, member_name in timestamped_members:
                if timestamp + 1e-9 < next_sample_time:
                    continue
                sampled_members.append(member_name)
                next_sample_time += sample_period
            duration = timestamped_members[-1][0] - start_time
            sequence_name = parent.split("/", 1)[0]
            tracks.append(
                VideoTrack(
                    archive_name=archive_name,
                    playlist_member=index_name,
                    label=f"{sequence_name} · {_duration_label(duration)}",
                    duration_seconds=duration,
                    rotation_degrees=0,
                    segment_members=tuple(sampled_members),
                    source_kind="tum_rgb",
                )
            )
        return tracks

    def track(self, archive_name: str, playlist_member: str) -> VideoTrack:
        for track in self.tracks(archive_name):
            if track.playlist_member == playlist_member:
                return track
        raise ValueError("unknown video track")


class FrameCache:
    def __init__(self, root: Path, ffmpeg: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ffmpeg = ffmpeg.expanduser().resolve()
        if not self.ffmpeg.is_file():
            raise FileNotFoundError(f"ffmpeg does not exist: {self.ffmpeg}")

    def prepare(
        self,
        archive_path: Path,
        track: VideoTrack,
        progress: Callable[[float, str], None],
    ) -> tuple[Path, ...]:
        target = self.root / track.cache_key
        metadata_path = target / "metadata.json"
        if metadata_path.is_file():
            frames = tuple(sorted((target / "frames").glob("*.jpg")))
            if frames:
                os.utime(metadata_path, None)
                progress(1.0, f"Ready · {len(frames)} sampled frames")
                return frames

        temporary = self.root / (
            f".{track.cache_key}.building-{os.getpid()}-{time.time_ns()}"
        )
        if temporary.exists():
            shutil.rmtree(temporary)
        frame_dir = temporary / "frames"
        frame_dir.mkdir(parents=True)

        if track.source_kind == "direct_video":
            command = [
                str(self.ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(archive_path),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-vf",
                f"fps={SOURCE_SAMPLE_HZ:g}",
                "-q:v",
                "3",
                "-start_number",
                "0",
                str(frame_dir / "%08d.jpg"),
            ]
            progress(0.1, "Decoding MP4 at 2 Hz")
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"ffmpeg exited with {result.returncode}: "
                        f"{result.stderr[-2000:]}"
                    )
                return self._commit_frames(temporary, target, track, progress)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise

        if track.source_kind == "tum_rgb":
            try:
                with tarfile.open(archive_path, mode="r:*") as archive:
                    member_by_name = {
                        member.name: member
                        for member in archive.getmembers()
                        if member.isfile()
                    }
                    total = len(track.segment_members)
                    for index, member_name in enumerate(track.segment_members):
                        source = archive.extractfile(member_by_name[member_name])
                        if source is None:
                            raise RuntimeError(f"cannot read RGB frame: {member_name}")
                        with source, Image.open(source) as image:
                            image.convert("RGB").save(
                                frame_dir / f"{index:08d}.jpg",
                                format="JPEG",
                                quality=90,
                            )
                        progress(
                            0.05 + 0.85 * ((index + 1) / total),
                            f"Reading TUM RGB sequence · frame {index + 1}/{total}",
                        )
                return self._commit_frames(temporary, target, track, progress)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        if track.source_kind != "hls_mpegts":
            shutil.rmtree(temporary, ignore_errors=True)
            raise ValueError(f"unsupported video source kind: {track.source_kind}")

        filters: list[str] = []
        if track.rotation_degrees == 90:
            filters.append("transpose=clock")
        elif track.rotation_degrees == 180:
            filters.extend(("hflip", "vflip"))
        elif track.rotation_degrees == 270:
            filters.append("transpose=cclock")
        filters.append(f"fps={SOURCE_SAMPLE_HZ:g}")
        command = [
            str(self.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "+genpts",
            "-f",
            "mpegts",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            ",".join(filters),
            "-q:v",
            "3",
            "-start_number",
            "0",
            str(frame_dir / "%08d.jpg"),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            if process.stdin is None or process.stderr is None:
                raise RuntimeError("failed to open ffmpeg pipes")
            with tarfile.open(archive_path, mode="r:") as archive:
                member_by_name = {
                    member.name: member
                    for member in archive.getmembers()
                    if member.isfile()
                }
                total = len(track.segment_members)
                for index, member_name in enumerate(track.segment_members):
                    source = archive.extractfile(member_by_name[member_name])
                    if source is None:
                        raise RuntimeError(f"cannot read segment: {member_name}")
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        process.stdin.write(chunk)
                    progress(
                        0.05 + 0.75 * ((index + 1) / total),
                        f"Decoding video · segment {index + 1}/{total}",
                    )
            process.stdin.close()
            error_text = process.stderr.read().decode("utf-8", errors="replace")
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"ffmpeg exited with {return_code}: {error_text[-2000:]}"
                )
            return self._commit_frames(temporary, target, track, progress)
        except Exception:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    @staticmethod
    def _commit_frames(
        temporary: Path,
        target: Path,
        track: VideoTrack,
        progress: Callable[[float, str], None],
    ) -> tuple[Path, ...]:
        frames = tuple(sorted((temporary / "frames").glob("*.jpg")))
        if not frames:
            raise RuntimeError("video source produced no sampled frames")
        metadata = {
            "archive": track.archive_name,
            "playlist": track.playlist_member,
            "source_kind": track.source_kind,
            "duration_seconds": track.duration_seconds,
            "source_sample_hz": SOURCE_SAMPLE_HZ,
            "frame_count": len(frames),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        if target.exists():
            shutil.rmtree(target)
        temporary.rename(target)
        final_frames = tuple(sorted((target / "frames").glob("*.jpg")))
        progress(1.0, f"Ready · {len(final_frames)} sampled frames")
        return final_frames


def _request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
    timeout: float = 180.0,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"model runtime HTTP {error.code}: {body}") from error


class VideoInferenceEngine:
    def __init__(
        self,
        catalog: ArchiveCatalog,
        cache: FrameCache,
        model_server_url: str,
        threshold: float,
    ) -> None:
        self.catalog = catalog
        self.cache = cache
        self.model_server_url = model_server_url.rstrip("/")
        self.threshold = threshold
        self.model_health = _request_json(
            self.model_server_url, "GET", "/health", timeout=10
        )
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="video-p1b-inference", daemon=True
        )
        self.track: VideoTrack | None = None
        self.frames: tuple[Path, ...] = ()
        self.loading = False
        self.loading_progress = 0.0
        self.direction = 0
        self.last_direction = 0
        self.next_index = 0
        self.last_inferred_index: int | None = None
        self.generation = 0
        self.segment_counter = 0
        self.segment_id = ""
        self.reset_needed = False
        self.frame_seq = 0
        self.image_version = 0
        self.status = "Select a ride"
        self.error: str | None = None
        self.rgb_jpeg: bytes | None = None
        self.semantic_png: bytes | None = None
        self.confidence_png: bytes | None = None
        self.gate_png: bytes | None = None
        self.observed_png: bytes | None = None
        self.guessed_png: bytes | None = None
        self.history_frame_count = 0
        self.inference_seconds: float | None = None
        self.depth_scale: float | None = None
        self.scale_std: float | None = None

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=10)

    def select(self, archive_name: str, playlist_member: str) -> None:
        track = self.catalog.track(archive_name, playlist_member)
        with self.condition:
            if self.loading:
                raise RuntimeError("wait for the current ride to finish loading")
            self.generation += 1
            generation = self.generation
            self.track = track
            self.frames = ()
            self.loading = True
            self.loading_progress = 0.0
            self.direction = 0
            self.last_direction = 0
            self.last_inferred_index = None
            self.next_index = 0
            self.reset_needed = False
            self.status = "Preparing video"
            self.error = None
            self.rgb_jpeg = None
            self.semantic_png = None
            self.confidence_png = None
            self.gate_png = None
            self.observed_png = None
            self.guessed_png = None
            self.frame_seq = 0
            self.image_version += 1
        threading.Thread(
            target=self._prepare,
            args=(generation, track),
            name="video-frame-cache",
            daemon=True,
        ).start()

    def _prepare(self, generation: int, track: VideoTrack) -> None:
        def progress(value: float, status: str) -> None:
            with self.condition:
                if generation == self.generation:
                    self.loading_progress = float(value)
                    self.status = status

        try:
            frames = self.cache.prepare(
                self.catalog.archive_path(track.archive_name), track, progress
            )
            preview = frames[0].read_bytes()
            with self.condition:
                if generation != self.generation:
                    return
                self.frames = frames
                self.loading = False
                self.loading_progress = 1.0
                self.rgb_jpeg = preview
                self.image_version += 1
                self.status = "Ready"
                self.condition.notify_all()
        except Exception as error:
            with self.condition:
                if generation == self.generation:
                    self.loading = False
                    self.status = "Video preparation failed"
                    self.error = str(error)

    def play(self, direction: int) -> None:
        if direction not in (-1, 1):
            raise ValueError("direction must be forward or reverse")
        with self.condition:
            if self.loading or not self.frames:
                raise RuntimeError("ride is not ready")
            if direction != self.last_direction:
                self.generation += 1
                self.segment_counter += 1
                self.segment_id = (
                    f"video-{self.track.cache_key}-{self.segment_counter}-"
                    f"{'forward' if direction > 0 else 'reverse'}"
                )
                self.reset_needed = True
                self.last_direction = direction
                if self.last_inferred_index is None:
                    self.next_index = 0 if direction > 0 else len(self.frames) - 1
                else:
                    candidate = self.last_inferred_index + direction
                    if candidate < 0:
                        candidate = len(self.frames) - 1
                    elif candidate >= len(self.frames):
                        candidate = 0
                    self.next_index = candidate
            elif self.next_index < 0 or self.next_index >= len(self.frames):
                self.next_index = 0 if direction > 0 else len(self.frames) - 1
            self.direction = direction
            self.status = "Playing forward" if direction > 0 else "Playing reverse"
            self.error = None
            self.condition.notify_all()

    def stop(self) -> None:
        with self.condition:
            self.direction = 0
            if self.frames:
                self.status = "Stopped"
            self.condition.notify_all()

    def replay(self) -> None:
        """Reset temporal state and immediately replay from the first frame."""
        with self.condition:
            if self.loading or not self.frames or self.track is None:
                raise RuntimeError("video is not ready")
            self.generation += 1
            self.segment_counter += 1
            self.segment_id = (
                f"video-{self.track.cache_key}-{self.segment_counter}-replay"
            )
            self.reset_needed = True
            self.direction = 1
            self.last_direction = 1
            self.next_index = 0
            self.last_inferred_index = None
            self.history_frame_count = 0
            self.status = "Replaying from start"
            self.error = None
            self.condition.notify_all()

    def snapshot(self) -> dict[str, object]:
        with self.condition:
            track = self.track
            index = self.last_inferred_index
            return {
                "ready": bool(self.frames) and not self.loading,
                "loading": self.loading,
                "loading_progress": self.loading_progress,
                "status": self.status,
                "error": self.error,
                "archive": None if track is None else track.archive_name,
                "track_id": None if track is None else track.playlist_member,
                "track_label": None if track is None else track.label,
                "direction": self.direction,
                "direction_name": (
                    "forward" if self.direction > 0
                    else "reverse" if self.direction < 0
                    else "stopped"
                ),
                "frame_index": index,
                "frame_count": len(self.frames),
                "source_time_seconds": (
                    None if index is None else index / SOURCE_SAMPLE_HZ
                ),
                "source_sample_hz": SOURCE_SAMPLE_HZ,
                "history_frame_count": self.history_frame_count,
                "max_history": MAX_HISTORY,
                "frame_seq": self.frame_seq,
                "image_version": self.image_version,
                "inference_seconds": self.inference_seconds,
                "depth_scale": self.depth_scale,
                "scale_std_m_per_vggt": self.scale_std,
                "model": self.model_health,
            }

    def image(self, kind: str) -> bytes | None:
        with self.condition:
            return {
                "rgb": self.rgb_jpeg,
                "semantic": self.semantic_png,
                "confidence": self.confidence_png,
                "gate": self.gate_png,
                "observed": self.observed_png,
                "guessed": self.guessed_png,
            }[kind]

    def _run(self) -> None:
        while not self.stop_event.is_set():
            with self.condition:
                while (
                    not self.stop_event.is_set()
                    and (self.direction == 0 or not self.frames)
                ):
                    self.condition.wait(timeout=0.5)
                if self.stop_event.is_set():
                    return
                direction = self.direction
                index = self.next_index
                frames = self.frames
                generation = self.generation
                segment_id = self.segment_id
                reset_needed = self.reset_needed
                self.reset_needed = False
            if not (0 <= index < len(frames)):
                with self.condition:
                    self.direction = 0
                    self.status = "End of ride"
                continue

            started = time.monotonic()
            try:
                if reset_needed:
                    _request_json(
                        self.model_server_url,
                        "POST",
                        "/reset",
                        {"segment_id": segment_id},
                        timeout=30,
                    )
                jpeg = frames[index].read_bytes()
                with Image.open(io.BytesIO(jpeg)) as image:
                    buffer = io.BytesIO()
                    image.convert("RGB").save(buffer, format="PNG")
                payload = {
                    "segment_id": segment_id,
                    "frame_seq": self.frame_seq + 1,
                    "image_png_base64": base64.b64encode(
                        buffer.getvalue()
                    ).decode("ascii"),
                    "threshold": self.threshold,
                }
                prediction = _request_json(
                    self.model_server_url,
                    "POST",
                    "/predict",
                    payload,
                    timeout=180,
                )
                semantic = base64.b64decode(
                    prediction["model_single_png_base64"], validate=True
                )
                if "model_merged_png_base64" in prediction:
                    confidence = base64.b64decode(
                        prediction["merged_score_png_base64"], validate=True
                    )
                    gate = semantic
                    observed = base64.b64decode(
                        prediction["model_merged_png_base64"], validate=True
                    )
                    guessed = base64.b64decode(
                        prediction["merged_score_png_base64"], validate=True
                    )
                else:
                    confidence = base64.b64decode(
                        prediction["confidence_single_png_base64"], validate=True
                    )
                    gate = base64.b64decode(
                        prediction["observed_gate_single_png_base64"], validate=True
                    )
                    observed = base64.b64decode(
                        prediction["observed_gate_confidence_png_base64"],
                        validate=True,
                    )
                    guessed = base64.b64decode(
                        prediction["guessed_occupancy_confidence_png_base64"],
                        validate=True,
                    )
            except Exception as error:
                with self.condition:
                    if generation == self.generation:
                        self.direction = 0
                        self.status = "Inference failed"
                        self.error = str(error)
                continue

            elapsed = time.monotonic() - started
            with self.condition:
                if generation != self.generation or frames is not self.frames:
                    continue
                self.rgb_jpeg = jpeg
                self.semantic_png = semantic
                self.confidence_png = confidence
                self.gate_png = gate
                self.observed_png = observed
                self.guessed_png = guessed
                self.last_inferred_index = index
                self.frame_seq += 1
                self.image_version += 1
                self.history_frame_count = int(
                    prediction["history_frame_count"]
                )
                self.inference_seconds = float(prediction["inference_seconds"])
                self.depth_scale = float(prediction["depth_scale"])
                self.scale_std = float(prediction["scale_std_m_per_vggt"])
                following = index + direction
                if following < 0 or following >= len(frames):
                    self.direction = 0
                    self.next_index = following
                    self.status = "End of ride"
                else:
                    self.next_index = following
                    if self.direction != 0:
                        self.status = (
                            "Playing forward" if direction > 0
                            else "Playing reverse"
                        )
                self.condition.notify_all()

            remaining = (1.0 / SOURCE_SAMPLE_HZ) - elapsed
            if remaining > 0:
                self.stop_event.wait(remaining)


def make_handler(
    engine: VideoInferenceEngine,
    html: bytes,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "P1BVideoVisualizer/1.0"

        def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _bytes(self, payload: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                if path == "/":
                    self._bytes(html, "text/html; charset=utf-8")
                elif path == "/api/archives":
                    self._json({"archives": engine.catalog.archive_list()})
                elif path == "/api/tracks":
                    archive_name = parse_qs(parsed.query).get("archive", [""])[0]
                    tracks = engine.catalog.tracks(archive_name)
                    self._json({"tracks": [track.public() for track in tracks]})
                elif path == "/api/state":
                    self._json(engine.snapshot())
                elif path.startswith("/api/image/"):
                    name = path.removeprefix("/api/image/")
                    mapping = {
                        "rgb.jpg": ("rgb", "image/jpeg"),
                        "semantic.png": ("semantic", "image/png"),
                        "confidence.png": ("confidence", "image/png"),
                        "gate.png": ("gate", "image/png"),
                        "observed.png": ("observed", "image/png"),
                        "guessed.png": ("guessed", "image/png"),
                    }
                    if name not in mapping:
                        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                        return
                    kind, content_type = mapping[name]
                    image = engine.image(kind)
                    if image is None:
                        self._json(
                            {"error": "image not ready"},
                            HTTPStatus.SERVICE_UNAVAILABLE,
                        )
                    else:
                        self._bytes(image, content_type)
                elif path == "/favicon.ico":
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1024 * 1024:
                    raise ValueError("request is too large")
                payload = json.loads(self.rfile.read(length) or b"{}")
                path = urlparse(self.path).path
                if path == "/api/select":
                    engine.select(
                        str(payload["archive"]), str(payload["track_id"])
                    )
                elif path == "/api/play":
                    direction_name = str(payload["direction"])
                    if direction_name == "forward":
                        engine.play(1)
                    elif direction_name == "reverse":
                        engine.play(-1)
                    else:
                        raise ValueError("direction must be forward or reverse")
                elif path == "/api/stop":
                    engine.stop()
                elif path == "/api/replay":
                    engine.replay()
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._json({"accepted": True})
            except Exception as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format_string: str, *args: object) -> None:
            request_line = str(args[0]) if args else ""
            if "/api/state" not in request_line:
                super().log_message(format_string, *args)

    return Handler


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    catalog = ArchiveCatalog(args.data_dir)
    cache = FrameCache(args.cache_dir, args.ffmpeg)
    html = args.html.expanduser().resolve().read_bytes()
    engine = VideoInferenceEngine(
        catalog,
        cache,
        args.model_server_url,
        args.threshold,
    )
    engine.start()
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(engine, html)
    )
    server.daemon_threads = True
    print(f"P1B video visualizer ready: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()


if __name__ == "__main__":
    main()
