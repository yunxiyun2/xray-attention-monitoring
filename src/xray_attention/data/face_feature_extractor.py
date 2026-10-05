"""面部特征提取器。

基于 MediaPipe Face Mesh（468 点 3D）提取眼/口/头姿/AU 等行为特征。
对应 spec v2.0 §4.2 面部特征。

几何参考：
- EAR: Soukupová & Čech (2016) Real-Time Eye Blink Detection Using Facial Landmarks.
- PERCLOS: NHTSA/Dinges 金标准。
- MAR: 与 EAR 同构，用于哈欠检测。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np


# ---------- 几何函数 ----------

def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """Eye Aspect Ratio。

    Parameters
    ----------
    eye_landmarks : 6 个 2D/3D 点，顺序：
        p1=左眼角, p2=上左, p3=上右, p4=右眼角, p5=下右, p6=下左

    Returns
    -------
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    闭眼时 → 0，睁眼时 ~0.2-0.4
    """
    pts = np.asarray(eye_landmarks, dtype=float)
    if pts.shape != (6, 2) and pts.shape != (6, 3):
        return 0.0
    p1, p2, p3, p4, p5, p6 = pts
    v1 = float(np.linalg.norm(p2 - p6))
    v2 = float(np.linalg.norm(p3 - p5))
    h = float(np.linalg.norm(p1 - p4))
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0


def calculate_mar(mouth_landmarks: np.ndarray) -> float:
    """Mouth Aspect Ratio（与 EAR 同构）。

    Parameters
    ----------
    mouth_landmarks : 6 点，顺序同 EAR：
        p1=左嘴角, p2=上左唇, p3=上右唇, p4=右嘴角, p5=下右唇, p6=下左唇

    Returns
    -------
    MAR，张嘴时大，闭嘴时 ≈ 0
    """
    return calculate_ear(mouth_landmarks)  # 公式同构


def _au43_proxy(eye_landmarks: np.ndarray, ear_open: float = 0.3) -> float:
    """AU43（眼睛闭合）的近似强度。

    用 1 - EAR/ear_open 截断到 [0, 1]。
    ear_open 为参考的「完全睁眼」EAR，默认 0.3。
    """
    ear = calculate_ear(eye_landmarks)
    val = 1.0 - ear / ear_open if ear_open > 0 else 1.0
    return float(np.clip(val, 0.0, 1.0))


# ---------- 头部姿态 ----------

def head_pose_from_landmarks(
    landmarks_3d: np.ndarray,
    image_size: tuple[int, int],
    camera_matrix: Optional[np.ndarray] = None,
) -> tuple[float, float, float]:
    """从 3D 关键点用 solvePnP 估计头部欧拉角。

    Parameters
    ----------
    landmarks_3d : MediaPipe FaceMesh 输出的 468×3 关键点（已含深度）
    image_size : (width, height)
    camera_matrix : 可选；缺省用单位矩阵

    Returns
    -------
    (yaw, pitch, roll) 单位：度
    """
    import cv2

    # 通用 3D 头部模型点（归一化）
    model_points = np.array([
        (0.0, 0.0, 0.0),          # 鼻尖
        (0.0, -63.6, -12.5),      # 下巴
        (-43.3, 32.7, -26.0),     # 左眼外角
        (43.3, 32.7, -26.0),      # 右眼外角
        (-28.9, -28.9, -24.1),    # 左嘴角
        (28.9, -28.9, -24.1),     # 右嘴角
    ], dtype=float)

    # 对应 MediaPipe 索引
    face_idx = [1, 152, 33, 263, 61, 291]
    image_points = np.array(
        [(landmarks_3d[i, 0], landmarks_3d[i, 1]) for i in face_idx], dtype=float
    )

    w, h = image_size
    if camera_matrix is None:
        camera_matrix = np.array([
            [w, 0, w / 2],
            [0, w, h / 2],
            [0, 0, 1.0],
        ], dtype=float)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    # 旋转矩阵 → 欧拉角
    proj = np.hstack([rmat, np.zeros((3, 1))])
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
    pitch, yaw, roll = float(euler[0]), float(euler[1]), float(euler[2])
    return yaw, pitch, roll


# ---------- 面部特征提取器 ----------

class FaceFeatureExtractor:
    """从 EAR/MAR 时序或视频帧计算窗口级面部行为特征。"""

    def perclos(self, ear_series: np.ndarray, ear_threshold: float = 0.1) -> float:
        """PERCLOS：EAR < 阈值的帧占比（金标准疲劳特征）。"""
        ear_series = np.asarray(ear_series, dtype=float)
        return float(np.mean(ear_series < ear_threshold))

    def long_closure_count(
        self, ear_series: np.ndarray, ear_threshold: float, min_frames: int, fps: float
    ) -> int:
        """单次闭合持续 >= min_frames/fps 秒的次数（micro-sleep 候选）。"""
        below = np.asarray(ear_series, dtype=float) < ear_threshold
        cnt = run = 0
        for v in below:
            run = run + 1 if v else 0
            if run == min_frames:
                cnt += 1
        return int(cnt)

    def detect_blinks(
        self, ear_series: np.ndarray, ear_threshold: float = 0.1
    ) -> list[dict]:
        """检测眨眼事件。

        眨眼 = EAR 从高降到阈值以下再回升。返回每个事件起止帧、持续帧数、幅度。
        """
        ear_series = np.asarray(ear_series, dtype=float)
        below = ear_series < ear_threshold
        # 找连续闭眼段
        diff = np.diff(below.astype(int))
        starts = np.where(diff == 1)[0] + 1   # 上升沿（睁→闭）的闭眼起点
        ends = np.where(diff == -1)[0] + 1    # 下降沿（闭→睁）的闭眼终点
        # 边界处理
        if below[0]:
            starts = np.r_[0, starts]
        if below[-1]:
            ends = np.r_[ends, len(below)]
        blinks = []
        for s, e in zip(starts, ends):
            duration = int(e - s)
            amplitude = float(ear_series.max() - ear_series[s:e].mean())
            blinks.append({
                "start": int(s), "end": int(e),
                "duration_frames": duration, "amplitude": amplitude,
            })
        return blinks

    def extract_window_features(
        self, ear_series: np.ndarray, ear_threshold: float, fps: float,
        mar_series: Optional[np.ndarray] = None,
        au43_series: Optional[np.ndarray] = None,
        pupil_series: Optional[np.ndarray] = None,
        head_yaw_series: Optional[np.ndarray] = None,
        head_pitch_series: Optional[np.ndarray] = None,
        head_roll_series: Optional[np.ndarray] = None,
    ) -> dict[str, float | int]:
        """对一段 EAR（+可选 MAR/AU43/瞳孔/头部姿态）时序提取窗口级特征。

        Parameters
        ----------
        ear_series : 一段 EAR 时序（窗口内逐帧）
        ear_threshold : PERCLOS / blink 判定阈值
        fps : 采样率
        mar_series : 可选，同长度 MAR 时序
        au43_series : 可选，AU43 闭眼代理强度时序（spec §4.2 AU）
        pupil_series : 可选，瞳孔直径时序（spec §4.2 瞳孔，如分辨率允许）
        head_yaw/pitch/roll_series : 可选，头部欧拉角时序（spec §4.2 头部姿态）

        Returns
        -------
        dict 含 perclos / long_closure_count / blink 统计 / ear 统计 / mar 统计
              / au43 统计 / pupil 统计 / head_pose 统计
        """
        ear = np.asarray(ear_series, dtype=float)
        blinks = self.detect_blinks(ear, ear_threshold)
        # 眨眼 = 持续 < 0.5s（即 < 0.5*fps 帧）的闭眼事件
        blink_thresh = int(0.5 * fps)
        real_blinks = [b for b in blinks if b["duration_frames"] < blink_thresh]
        window_sec = len(ear) / fps if fps > 0 else 0
        blink_rate = len(real_blinks) / window_sec * 60 if window_sec > 0 else 0.0

        feat: dict[str, float | int] = {
            "perclos_80": self.perclos(ear, ear_threshold),
            "long_closure_count": self.long_closure_count(
                ear, ear_threshold, min_frames=int(fps), fps=fps  # >=1s 为 micro-sleep
            ),
            "blink_count": len(real_blinks),
            "blink_rate": float(blink_rate),
            "blink_duration_mean": (
                float(np.mean([b["duration_frames"] for b in real_blinks]))
                if real_blinks else 0.0
            ),
            "blink_amplitude_mean": (
                float(np.mean([b["amplitude"] for b in real_blinks]))
                if real_blinks else 0.0
            ),
            "ear_mean": float(np.mean(ear)),
            "ear_std": float(np.std(ear)),
            "ear_min": float(np.min(ear)),
        }
        if mar_series is not None:
            mar = np.asarray(mar_series, dtype=float)
            yawn_events = np.sum(
                np.diff((mar > 0.5).astype(int)) == 1
            )
            feat.update({
                "mar_mean": float(np.mean(mar)),
                "mar_max": float(np.max(mar)),
                "yawn_count": int(yawn_events),
            })
        # AU43 闭眼代理（spec §4.2：若 OpenFace 不可用则用 FaceMesh 近似）
        if au43_series is not None:
            au = np.asarray(au43_series, dtype=float)
            if len(au) > 0:
                feat.update({
                    "au43_mean": float(np.mean(au)),
                    "au43_ratio": float(np.mean(au > 0.5)),
                })
        # 瞳孔直径（spec §4.2：如分辨率允许）
        if pupil_series is not None:
            pup = np.asarray(pupil_series, dtype=float)
            if len(pup) > 0:
                feat.update({
                    "pupil_diameter_mean": float(np.mean(pup)),
                    "pupil_diameter_std": float(np.std(pup)),
                })
        # 头部姿态（spec §4.2：yaw/pitch/roll + 点头次数）
        if head_pitch_series is not None:
            pitch = np.asarray(head_pitch_series, dtype=float)
            if len(pitch) > 0:
                # 点头 = pitch 周期性负峰值（下低头）
                neg_peaks = np.sum(
                    np.diff((np.diff(pitch) > 0).astype(int)) == -1
                )
                feat["head_nod_count"] = int(neg_peaks)
                feat["head_pitch_std"] = float(np.std(pitch))
        if head_yaw_series is not None:
            yaw = np.asarray(head_yaw_series, dtype=float)
            if len(yaw) > 0:
                feat["head_yaw_std"] = float(np.std(yaw))
        if head_roll_series is not None:
            roll = np.asarray(head_roll_series, dtype=float)
            if len(roll) > 0:
                feat["head_roll_std"] = float(np.std(roll))
        return feat


# ---------- MediaPipe 视频流水线 ----------

# MediaPipe FaceMesh 关键点索引
_LM_LEFT_EYE = [33, 160, 158, 133, 153, 144]      # p1..p6
_LM_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
_LM_MOUTH = [61, 39, 0, 291, 17, 18]
_LM_LEFT_IRIS = [468]   # refine_landmarks 时
_LM_RIGHT_IRIS = [473]


def extract_from_video(
    video_path: str | Path,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
) -> dict[str, np.ndarray]:
    """用 MediaPipe FaceMesh 逐帧提取面部时序。

    Parameters
    ----------
    video_path : 视频文件路径
    max_frames : 可选最大处理帧数（调试用）
    frame_stride : 帧子采样步长（>1 时每 stride 帧取一帧，
        有效采样率 = fps / stride）。PERCLOS 在 10fps 下仍可靠。

    Returns
    -------
    dict 含 left_ear / right_ear / avg_ear / mar / head_yaw / head_pitch / head_roll /
          au43_left / au43_right / pupil_diameter 序列，以及 fps（有效采样率）
    """
    import cv2
    import mediapipe as mp

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    raw_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows: list[dict] = []
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as mesh:
        idx = 0
        processed = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            # 子采样：只处理每 stride 帧的第 1 帧
            if idx % frame_stride == 0:
                if max_frames is not None and processed >= max_frames:
                    break
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = mesh.process(rgb)
                row = _empty_face_row()
                if res.multi_face_landmarks:
                    lms = np.array(
                        [(lm.x * w, lm.y * h, lm.z * w) for lm in res.multi_face_landmarks[0].landmark]
                    )
                    le = lms[_LM_LEFT_EYE]
                    re = lms[_LM_RIGHT_EYE]
                    mt = lms[_LM_MOUTH]
                    row["left_ear"] = calculate_ear(le)
                    row["right_ear"] = calculate_ear(re)
                    row["avg_ear"] = (row["left_ear"] + row["right_ear"]) / 2
                    row["mar"] = calculate_mar(mt)
                    row["au43_left"] = _au43_proxy(le)
                    row["au43_right"] = _au43_proxy(re)
                    try:
                        yaw, pitch, roll = head_pose_from_landmarks(lms, (w, h))
                        row["head_yaw"], row["head_pitch"], row["head_roll"] = yaw, pitch, roll
                    except Exception:
                        pass
                    if len(lms) > _LM_RIGHT_IRIS[0]:
                        row["pupil_diameter"] = _pupil_diameter(lms, _LM_LEFT_IRIS, _LM_RIGHT_IRIS)
                rows.append(row)
                processed += 1
            idx += 1
    cap.release()

    if not rows:
        return {"fps": raw_fps / frame_stride}
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["fps"] = raw_fps / frame_stride
    return out


def _empty_face_row() -> dict:
    return {
        "left_ear": 0.0, "right_ear": 0.0, "avg_ear": 0.0, "mar": 0.0,
        "head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0,
        "au43_left": 0.0, "au43_right": 0.0, "pupil_diameter": 0.0,
    }


def _pupil_diameter(lms: np.ndarray, left_idx: list[int], right_idx: list[int]) -> float:
    """用虹膜关键点的外接圆近似直径。MediaPipe refine_landmarks 提供 5 点/虹膜。"""
    # 468-477 为左右虹膜各 5 点
    left_iris = lms[468:473]
    right_iris = lms[473:478]
    d_left = float(np.linalg.norm(left_iris.max(0) - left_iris.min(0)))
    d_right = float(np.linalg.norm(right_iris.max(0) - right_iris.min(0)))
    return (d_left + d_right) / 2.0
