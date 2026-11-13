"""MediaPipe keypoint extraction for isolated ISL signs.

Per frame we keep:

* 25 upper-body pose points (MediaPipe pose indices 0-24), x/y
* 21 landmarks for each hand, x/y
* 3 presence flags: pose, left hand, right hand

All coordinates are made shoulder-relative: subtract the shoulder midpoint and
divide by shoulder width. This removes camera distance and image position, so
the model sees body/hand shape rather than where the signer stands.

Hands are assigned to left/right by distance to the pose wrists, not by
MediaPipe's handedness label, which is ambiguous for mirrored webcams.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

N_POSE = 25
N_HAND = 21
L_SHOULDER = 11
R_SHOULDER = 12
L_WRIST = 15
R_WRIST = 16


@dataclass
class FrameRaw:
    pose: list | None
    left_hand: list | None
    right_hand: list | None


class Landmarkers:
    """Reusable pose + hand landmarkers (VIDEO mode)."""

    def __init__(
        self,
        hand_model: Path,
        pose_model: Path,
        *,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(hand_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(pose_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )
        # MediaPipe VIDEO mode requires strictly increasing timestamps for the
        # whole lifetime of the landmarker, not just within one video. Each new
        # clip therefore continues after the previous clip's last timestamp.
        self.timestamp_cursor = 0

    @classmethod
    def from_config(cls, cfg) -> "Landmarkers":
        models = cfg.path("keypoints.models_dir")
        return cls(
            models / str(cfg.get("keypoints.models.hand.file")),
            models / str(cfg.get("keypoints.models.pose.file")),
            num_hands=int(cfg.get("keypoints.num_hands", 2)),
            min_detection_confidence=float(
                cfg.get("keypoints.min_detection_confidence", 0.5)
            ),
            min_tracking_confidence=float(
                cfg.get("keypoints.min_tracking_confidence", 0.5)
            ),
        )

    def close(self) -> None:
        self.hand.close()
        self.pose.close()

    def __enter__(self) -> "Landmarkers":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _xy(lm, use_z: bool) -> np.ndarray:
    if use_z:
        return np.array([lm.x, lm.y, lm.z], dtype=np.float32)
    return np.array([lm.x, lm.y], dtype=np.float32)


def _visible(lm, threshold: float) -> bool:
    visibility = float(getattr(lm, "visibility", 1.0))
    presence = float(getattr(lm, "presence", 1.0))
    return visibility >= threshold and presence >= threshold


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def timestamp_at(base: int, index: int, fps: float) -> int:
    """Timestamp for frame ``index`` of a clip starting at ``base`` (ms)."""
    return base + int(round(index * 1000.0 / fps))


def _shoulder_normalisation(
    pose: list | None, threshold: float
) -> tuple[np.ndarray, float] | None:
    if pose is None or len(pose) <= R_SHOULDER:
        return None
    ls, rs = pose[L_SHOULDER], pose[R_SHOULDER]
    if not (_visible(ls, threshold) and _visible(rs, threshold)):
        return None
    centre = (_xy(ls, False) + _xy(rs, False)) / 2.0
    scale = float(np.linalg.norm(_xy(ls, False) - _xy(rs, False)))
    if scale < 1e-4:
        return None
    return centre, scale


def _handedness_name(handedness, index: int) -> str:
    try:
        return str(handedness[index][0].category_name).lower()
    except (IndexError, AttributeError, TypeError):
        return ""


def assign_hands(
    hand_landmarks: list,
    handedness: list,
    pose: list | None,
    threshold: float,
) -> tuple[list | None, list | None]:
    """Return (left_hand, right_hand), using pose wrists when available."""
    if not hand_landmarks:
        return None, None

    wrists: dict[str, np.ndarray] = {}
    if pose is not None and len(pose) > R_WRIST:
        if _visible(pose[L_WRIST], threshold):
            wrists["left"] = _xy(pose[L_WRIST], False)
        if _visible(pose[R_WRIST], threshold):
            wrists["right"] = _xy(pose[R_WRIST], False)

    if len(hand_landmarks) == 1:
        hand = hand_landmarks[0]
        if wrists:
            tip = _xy(hand[0], False)
            side = min(wrists, key=lambda name: _dist(tip, wrists[name]))
        else:
            side = "left" if _handedness_name(handedness, 0) == "left" else "right"
        return (hand, None) if side == "left" else (None, hand)

    first, second = hand_landmarks[0], hand_landmarks[1]
    if len(wrists) == 2:
        same = _dist(_xy(first[0], False), wrists["left"]) + _dist(
            _xy(second[0], False), wrists["right"]
        )
        cross = _dist(_xy(first[0], False), wrists["right"]) + _dist(
            _xy(second[0], False), wrists["left"]
        )
        if cross < same:
            first, second = second, first
        return first, second

    # No usable wrists: fall back to MediaPipe's labels, then detection order.
    slots: list[list | None] = [None, None]
    for i, hand in enumerate((first, second)):
        side = _handedness_name(handedness, i)
        idx = 0 if side == "left" else (1 if side == "right" else i)
        if slots[idx] is None:
            slots[idx] = hand
        else:
            slots[1 - idx] = hand
    return slots[0], slots[1]


def detect_frame(
    landmarkers: Landmarkers,
    image,
    timestamp_ms: int,
    shoulder_visibility: float,
) -> FrameRaw:
    """Run pose + hands on one already-converted ``mp.Image``."""
    pose_result = landmarkers.pose.detect_for_video(image, timestamp_ms)
    hand_result = landmarkers.hand.detect_for_video(image, timestamp_ms)

    pose = list(pose_result.pose_landmarks[0]) if pose_result.pose_landmarks else None
    left, right = assign_hands(
        list(hand_result.hand_landmarks),
        list(hand_result.handedness),
        pose,
        shoulder_visibility,
    )
    return FrameRaw(pose=pose, left_hand=left, right_hand=right)


def read_video_frames(
    landmarkers: Landmarkers,
    video_path: Path,
    *,
    shoulder_visibility: float = 0.3,
) -> tuple[list[FrameRaw], float, int]:
    """Run MediaPipe on every video frame and keep raw landmarks."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    frames: list[FrameRaw] = []
    base = landmarkers.timestamp_cursor
    last_ts = base - 1
    try:
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp = timestamp_at(base, len(frames), fps)
            if timestamp <= last_ts:
                timestamp = last_ts + 1
            last_ts = timestamp

            frames.append(
                detect_frame(landmarkers, image, timestamp, shoulder_visibility)
            )
    finally:
        cap.release()
        # Advance even on failure, so the next clip never reuses a timestamp.
        landmarkers.timestamp_cursor = last_ts + 1

    if not frames:
        raise RuntimeError(f"No decodable frames: {video_path}")
    return frames, fps, len(frames)


def _fill_points(
    out: np.ndarray,
    offset: int,
    landmarks: list | None,
    count: int,
    centre: np.ndarray,
    scale: float,
    channels: int,
) -> int:
    if landmarks is None:
        return offset
    for i, lm in enumerate(landmarks[:count]):
        point = _xy(lm, channels == 3)
        point[0] = (point[0] - centre[0]) / scale
        point[1] = (point[1] - centre[1]) / scale
        if channels == 3:
            point[2] = point[2] / scale
        out[offset + i * channels : offset + (i + 1) * channels] = point
    return offset + count * channels


def assemble_features(
    frames: list[FrameRaw],
    *,
    pose_points: int = N_POSE,
    use_z: bool = False,
    presence_flags: bool = True,
    shoulder_visibility: float = 0.3,
) -> np.ndarray:
    """Convert raw landmarks to normalised per-frame feature vectors."""
    channels = 3 if use_z else 2
    dim = pose_points * channels + 2 * N_HAND * channels
    if presence_flags:
        dim += 3
    out = np.zeros((len(frames), dim), dtype=np.float32)

    # Stable fallback for clips where the shoulders are briefly invisible:
    # use the median shoulder frame that did work.
    valid = [
        norm
        for frame in frames
        if (norm := _shoulder_normalisation(frame.pose, shoulder_visibility))
        is not None
    ]
    if valid:
        fallback_centre = np.median([v[0] for v in valid], axis=0)
        fallback_scale = float(np.median([v[1] for v in valid]))
    else:
        fallback_centre = np.array([0.5, 0.5], dtype=np.float32)
        fallback_scale = 0.25

    last_norm = (fallback_centre, fallback_scale)

    for t, frame in enumerate(frames):
        norm = _shoulder_normalisation(frame.pose, shoulder_visibility)
        if norm is not None:
            last_norm = norm
        centre, scale = last_norm

        offset = 0
        pose_present = norm is not None
        if frame.pose is not None:
            offset = _fill_points(
                out[t],
                offset,
                frame.pose,
                pose_points,
                centre,
                scale,
                channels,
            )

        left_present = frame.left_hand is not None
        right_present = frame.right_hand is not None
        offset = _fill_points(
            out[t], offset, frame.left_hand, N_HAND, centre, scale, channels
        )
        offset = _fill_points(
            out[t], offset, frame.right_hand, N_HAND, centre, scale, channels
        )

        if presence_flags:
            out[t, -3:] = (float(pose_present), float(left_present), float(right_present))

    return out


def resample_sequence(features: np.ndarray, num_frames: int, *, flags: int = 3) -> np.ndarray:
    """Linearly resample a (T, D) sequence to (num_frames, D)."""
    if features.shape[0] == num_frames:
        result = features.copy()
    elif features.shape[0] == 1:
        result = np.repeat(features, num_frames, axis=0)
    else:
        src = np.linspace(0, features.shape[0] - 1, features.shape[0])
        dst = np.linspace(0, features.shape[0] - 1, num_frames)
        result = np.empty((num_frames, features.shape[1]), dtype=np.float32)
        for column in range(features.shape[1]):
            result[:, column] = np.interp(
                dst, src, features[:, column]
            ).astype(np.float32)

    if flags:
        result[:, -flags:] = (result[:, -flags:] >= 0.5).astype(np.float32)
    return result


def extract_video(
    landmarkers: Landmarkers,
    video_path: Path,
    *,
    pose_points: int = N_POSE,
    use_z: bool = False,
    presence_flags: bool = True,
    resample_frames: int = 32,
    shoulder_visibility: float = 0.3,
) -> tuple[np.ndarray, dict]:
    """Return (features[num_frames, D], metadata) for one video."""
    frames, fps, n_frames = read_video_frames(
        landmarkers,
        video_path,
        shoulder_visibility=shoulder_visibility,
    )
    per_frame = assemble_features(
        frames,
        pose_points=pose_points,
        use_z=use_z,
        presence_flags=presence_flags,
        shoulder_visibility=shoulder_visibility,
    )
    features = resample_sequence(
        per_frame, resample_frames, flags=3 if presence_flags else 0
    )
    meta = {
        "fps": fps,
        "n_frames": n_frames,
        "resampled": resample_frames,
        "pose_frames": int(per_frame[:, -3].sum()) if presence_flags else 0,
        "hand_frames": int(per_frame[:, -2:].sum()) if presence_flags else 0,
    }
    return features, meta
