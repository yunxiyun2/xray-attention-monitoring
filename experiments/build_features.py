"""一键构建注视/面部/多模态特征矩阵。

对应 spec v2.0 §12 文件结构 experiments/build_features.py。

用法（从项目根目录运行）：
    PYTHONPATH=src conda run -n xray-attention python experiments/build_features.py \
        --config configs/multimodal.yaml

可选参数：
    --with-face : 同时从视频提取面部特征（耗时较长）
    --max-frames N : 调试时限制每个视频处理帧数
    --frame-stride K : 帧子采样步长（K=3 时 ~30fps→10fps，PERCLOS 仍可靠）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from xray_attention.data.dataset_builder import DatasetBuilder
from xray_attention.data.face_feature_extractor import FaceFeatureExtractor

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="构建多模态特征矩阵")
    parser.add_argument("--config", default="configs/multimodal.yaml")
    parser.add_argument("--with-face", action="store_true", help="同时提取面部特征")
    parser.add_argument("--max-frames", type=int, default=None, help="调试限帧")
    parser.add_argument(
        "--frame-stride", type=int, default=3,
        help="帧子采样步长（默认 3，即 30fps→10fps）",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(_PROJECT_ROOT / args.config))
    window_sec = cfg["window_sec"]
    stride_sec = cfg["stride_train_sec"]  # 训练用 50% 重叠
    thresholds = cfg["threshold"]
    fps = 30.0

    builder = DatasetBuilder()
    records = builder.discover_records()
    print(f"[1/4] 发现 {len(records)} 条 record")

    # ---------- 注视特征 ----------
    print(f"[2/4] 提取注视特征（窗口={window_sec}s, 步长={stride_sec}s）...")
    gaze_df = builder.build_gaze_feature_matrix(
        window_sec=window_sec,
        stride_sec=stride_sec,
        fps=fps,
        thresholds=thresholds,
    )
    out_dir = _PROJECT_ROOT / "dataset" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    gaze_csv = out_dir / "gaze_features.csv"
    gaze_df.to_csv(gaze_csv, index=False)
    print(f"      → {gaze_csv}  ({len(gaze_df)} 行, {len(gaze_df.columns)} 列)")

    # ---------- 面部特征 ----------
    face_csv = out_dir / "face_features.csv"
    if args.with_face:
        print(
            f"[3/4] 提取面部特征（MediaPipe，限帧={args.max_frames}, "
            f"stride={args.frame_stride}）..."
        )
        face_ext = FaceFeatureExtractor()
        face_rows: list[dict] = []
        done = 0
        for rec in records:
            if not rec.video_path.exists():
                continue
            try:
                from xray_attention.data.face_feature_extractor import extract_from_video
                series = extract_from_video(
                    rec.video_path,
                    max_frames=args.max_frames,
                    frame_stride=args.frame_stride,
                )
                ear = series.get("avg_ear", np.array([]))
                mar = series.get("mar", np.array([]))
                au43 = series.get("au43_left", np.array([]))
                pupil = series.get("pupil_diameter", np.array([]))
                hyaw = series.get("head_yaw", np.array([]))
                hpitch = series.get("head_pitch", np.array([]))
                hroll = series.get("head_roll", np.array([]))
                if len(ear) == 0:
                    continue
                # 使用返回的有效采样率（raw_fps / stride）做窗口切分
                eff_fps = float(series.get("fps", fps / args.frame_stride))
                win_frames = int(window_sec * eff_fps)
                step_frames = int(stride_sec * eff_fps)
                if win_frames < 1:
                    continue
                for s in range(0, len(ear) - win_frames + 1, max(step_frames, 1)):
                    seg = ear[s : s + win_frames]
                    seg_mar = (
                        mar[s : s + win_frames]
                        if len(mar) >= s + win_frames else None
                    )
                    seg_au43 = (
                        au43[s : s + win_frames]
                        if len(au43) >= s + win_frames else None
                    )
                    seg_pupil = (
                        pupil[s : s + win_frames]
                        if len(pupil) >= s + win_frames else None
                    )
                    seg_hyaw = (
                        hyaw[s : s + win_frames]
                        if len(hyaw) >= s + win_frames else None
                    )
                    seg_hpitch = (
                        hpitch[s : s + win_frames]
                        if len(hpitch) >= s + win_frames else None
                    )
                    seg_hroll = (
                        hroll[s : s + win_frames]
                        if len(hroll) >= s + win_frames else None
                    )
                    feat = face_ext.extract_window_features(
                        seg, ear_threshold=0.1, fps=eff_fps,
                        mar_series=seg_mar, au43_series=seg_au43,
                        pupil_series=seg_pupil,
                        head_yaw_series=seg_hyaw,
                        head_pitch_series=seg_hpitch,
                        head_roll_series=seg_hroll,
                    )
                    feat.update({
                        "subject": rec.subject,
                        "label": 1 if rec.state == "alert" else 0,
                        "state": rec.state,
                        "difficulty": rec.difficulty,
                        "window_idx": s,
                    })
                    face_rows.append(feat)
                done += 1
                if done % 5 == 0:
                    print(f"      已处理 {done} 个视频")
            except Exception as e:
                print(f"      [警告] {rec.subject}/{rec.state}/{rec.difficulty} 失败: {e}")
                continue
        face_df = pd.DataFrame(face_rows)
        face_df.to_csv(face_csv, index=False)
        print(f"      → {face_csv}  ({len(face_df)} 行)")
    else:
        print("[3/4] 跳过面部特征提取（加 --with-face 启用）")

    # ---------- LOSO 划分 ----------
    print("[4/4] 写 LOSO 划分表...")
    builder.write_loso_folds(n_subjects=20)
    folds_csv = _PROJECT_ROOT / "dataset" / "splits" / "loso_folds.csv"
    print(f"      → {folds_csv}")

    print("\n完成。可用产物：")
    print(f"  注视特征: {gaze_csv}")
    if args.with_face:
        print(f"  面部特征: {face_csv}")
    print(f"  LOSO 划分: {folds_csv}")


if __name__ == "__main__":
    main()
