#!/usr/bin/env python3
"""Extract deterministic, temporally separated frames from one local clip."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, TextIO


class FrameExtractionError(RuntimeError):
    """Raised when a finite clip cannot be safely sampled."""


@dataclass(frozen=True, slots=True)
class ExtractedFrame:
    source_clip: str
    session: str
    frame_index: int
    timestamp_s: float
    width: int
    height: int
    image_file: str


@dataclass(frozen=True, slots=True)
class ExtractionOptions:
    source: Path
    output_dir: Path
    manifest: Path
    session: str
    sample_fps: float = 2.0
    minimum_frame_gap: int = 2
    jpeg_quality: int = 95
    dry_run: bool = False

    def __post_init__(self) -> None:
        if not self.source.is_file():
            raise ValueError(f"source clip does not exist: {self.source}")
        if not self.session or self.session != _safe_name(self.session):
            raise ValueError("session must contain only letters, digits, hyphens, or underscores")
        if not math.isfinite(self.sample_fps) or self.sample_fps <= 0.0:
            raise ValueError("sample-fps must be finite and greater than zero")
        if (
            isinstance(self.minimum_frame_gap, bool)
            or not isinstance(self.minimum_frame_gap, int)
            or self.minimum_frame_gap < 2
        ):
            raise ValueError("minimum-frame-gap must be an integer of at least 2")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg-quality must be between 1 and 100")
        if self.manifest.suffix.casefold() not in {".json", ".csv"}:
            raise ValueError("manifest must end in .json or .csv")


def extract_frames(
    options: ExtractionOptions, *, cv2_module: Any | None = None
) -> tuple[ExtractedFrame, ...]:
    """Decode and sample a clip; dry-run performs no filesystem writes."""
    cv2 = cv2_module if cv2_module is not None else _load_cv2()
    capture: Any | None = None
    records: list[ExtractedFrame] = []
    seen_digests: set[str] = set()
    last_selected_index: int | None = None
    try:
        capture = cv2.VideoCapture(str(options.source))
        if not bool(capture.isOpened()):
            raise FrameExtractionError(f"unable to open source clip {options.source}")
        source_fps = _capture_fps(capture, cv2)
        sample_interval_s = 1.0 / options.sample_fps
        next_sample_s = 0.0
        frame_index = 0
        while True:
            result = capture.read()
            if not isinstance(result, tuple) or len(result) != 2:
                raise FrameExtractionError("clip returned an invalid read result")
            ok, image = result
            if not ok:
                break
            if image is None:
                raise FrameExtractionError("clip returned no image for a successful read")
            height, width = _image_dimensions(image)
            timestamp_s = _frame_timestamp(capture, cv2, frame_index, source_fps)
            if timestamp_s + 1e-9 >= next_sample_s:
                while next_sample_s <= timestamp_s + 1e-9:
                    next_sample_s += sample_interval_s
                separated = (
                    last_selected_index is None
                    or frame_index - last_selected_index >= options.minimum_frame_gap
                )
                digest = _image_digest(image, width, height)
                if separated and digest not in seen_digests:
                    clip_identity = _safe_name(options.source.stem)
                    identity = (
                        options.session
                        if clip_identity == options.session
                        else f"{options.session}__{clip_identity}"
                    )
                    filename = (
                        f"{identity}__f{frame_index:08d}__t{round(timestamp_s * 1000):010d}.jpg"
                    )
                    destination = options.output_dir / filename
                    if destination.exists():
                        raise FrameExtractionError(
                            f"refusing to overwrite extracted frame {destination}"
                        )
                    if not options.dry_run:
                        options.output_dir.mkdir(parents=True, exist_ok=True)
                        written = cv2.imwrite(
                            str(destination),
                            image,
                            [int(cv2.IMWRITE_JPEG_QUALITY), options.jpeg_quality],
                        )
                        if not bool(written):
                            raise FrameExtractionError(
                                f"OpenCV failed to write extracted frame {destination}"
                            )
                    seen_digests.add(digest)
                    last_selected_index = frame_index
                    records.append(
                        ExtractedFrame(
                            source_clip=options.source.name,
                            session=options.session,
                            frame_index=frame_index,
                            timestamp_s=timestamp_s,
                            width=width,
                            height=height,
                            image_file=filename,
                        )
                    )
            frame_index += 1
    finally:
        if capture is not None:
            capture.release()

    if not records:
        raise FrameExtractionError("no frames satisfied the sampling policy")
    if not options.dry_run:
        _write_manifest(options.manifest, records)
    return tuple(records)


def _capture_fps(capture: Any, cv2: Any) -> float:
    try:
        value = float(capture.get(cv2.CAP_PROP_FPS))
    except Exception as exc:
        raise FrameExtractionError(f"unable to read source frame rate: {exc}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise FrameExtractionError("source clip does not report a usable frame rate")
    return value


def _frame_timestamp(
    capture: Any,
    cv2: Any,
    frame_index: int,
    source_fps: float,
) -> float:
    fallback = frame_index / source_fps
    try:
        position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    except Exception:
        return fallback
    if (
        not math.isfinite(position_ms)
        or position_ms < 0.0
        or (frame_index > 0 and position_ms == 0.0)
    ):
        return fallback
    return position_ms / 1000.0


def _image_dimensions(image: object) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if not isinstance(shape, tuple) or len(shape) not in {2, 3}:
        raise FrameExtractionError("decoded frame must have a 2D or 3D shape")
    height, width = shape[:2]
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height <= 0
        or width <= 0
    ):
        raise FrameExtractionError("decoded frame dimensions must be positive integers")
    return height, width


def _image_digest(image: object, width: int, height: int) -> str:
    tobytes = getattr(image, "tobytes", None)
    if not callable(tobytes):
        raise FrameExtractionError("decoded frame does not provide stable image bytes")
    payload = tobytes()
    if not isinstance(payload, bytes):
        raise FrameExtractionError("decoded frame bytes are invalid")
    digest = hashlib.sha256()
    digest.update(f"{width}x{height}\0".encode())
    digest.update(payload)
    return digest.hexdigest()


def _write_manifest(path: Path, records: list[ExtractedFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FrameExtractionError(f"refusing to overwrite manifest {path}")
    if path.suffix.casefold() == ".json":
        serialized = (
            json.dumps(
                {
                    "schema_version": 1,
                    "frames": [asdict(record) for record in records],
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        _atomic_text_write(path, serialized)
        return

    fieldnames = list(asdict(records[0]))

    def write_csv(stream: TextIO) -> None:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    _atomic_stream_write(path, write_csv)


def _atomic_text_write(path: Path, serialized: str) -> None:
    _atomic_stream_write(path, lambda stream: stream.write(serialized))


def _atomic_stream_write(path: Path, writer: Any) -> None:
    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            descriptor = -1
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_cv2() -> Any:
    try:
        return import_module("cv2")
    except (ImportError, ModuleNotFoundError) as exc:
        raise FrameExtractionError(
            "OpenCV is unavailable; install the constrained vision dependencies"
        ) from exc


def _safe_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_" for character in value
    )
    return cleaned.strip("_-") or "session"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session", help="recording-session id; defaults to clip stem")
    parser.add_argument("--manifest", type=Path, help="new .json or .csv manifest path")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--minimum-frame-gap", type=int, default=2)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    session = args.session or _safe_name(args.source.stem)
    manifest = args.manifest or args.output_dir.parent / f"{session}-manifest.json"
    try:
        options = ExtractionOptions(
            source=args.source,
            output_dir=args.output_dir,
            manifest=manifest,
            session=session,
            sample_fps=args.sample_fps,
            minimum_frame_gap=args.minimum_frame_gap,
            jpeg_quality=args.jpeg_quality,
            dry_run=args.dry_run,
        )
        records = extract_frames(options)
    except Exception as exc:
        print(f"extraction error: {exc}", file=sys.stderr)
        return 2
    mode = "dry-run candidates" if args.dry_run else "written frames"
    print(f"{mode}={len(records)} session={session} manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
