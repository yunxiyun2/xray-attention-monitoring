"""数据集构建与时间对齐。

对应 spec v2.0 §3 数据来源、§5 时间窗口与样本切分。

- discover_records : 扫描 Distan_error 目录，发现 20×2×2=80 条 record
- build_windows_for_record : 对单条 record 切窗（测试步长=窗口长度，不重叠）
- write_loso_folds : 生成 20 折 LOSO 划分表
- build_multimodal_features : 对齐注视与面部时序，输出 CSV 特征矩阵
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from xray_attention.data.gaze_feature_extractor import GazeFeatureExtractor

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TaskRecord:
    """一条受试者×状态×难度的记录。"""
    subject: str          # 受试者 ID，如 "01"
    state: str            # alert | sleepy
    difficulty: str       # easy | hard
    video_path: Path      # 视频文件路径（可能不存在）
    errors_path: Path     # 注视误差序列文件路径


class DatasetBuilder:
    """发现记录、切窗、对齐注视与面部特征。"""

    def __init__(
        self,
        data_root: Optional[str | Path] = None,
        dist_error_root: Optional[str | Path] = None,
        output_root: Optional[str | Path] = None,
    ):
        self.data_root = Path(data_root) if data_root else _PROJECT_ROOT / "data"
        self.dist_error_root = (
            Path(dist_error_root) if dist_error_root
            else _PROJECT_ROOT / "Distan_error（原20）"
        )
        self.output_root = Path(output_root) if output_root else _PROJECT_ROOT / "dataset"

    # ---------- 记录发现 ----------

    def discover_records(self) -> list[TaskRecord]:
        """扫描 dist_error_root，发现所有 <difficulty>/<subject>/<state>/all_errors.txt。"""
        records: list[TaskRecord] = []
        if not self.dist_error_root.exists():
            return records
        for diff_dir in sorted(self.dist_error_root.iterdir()):
            if not diff_dir.is_dir() or diff_dir.name not in ("easy", "hard"):
                continue
            for subj_dir in sorted(diff_dir.iterdir()):
                if not subj_dir.is_dir() or not subj_dir.name.isdigit():
                    continue
                for state_dir in sorted(subj_dir.iterdir()):
                    if state_dir.name not in ("alert", "sleepy"):
                        continue
                    err_file = state_dir / "all_errors.txt"
                    # 视频路径尝试多种约定
                    vid = self._find_video(subj_dir.name, state_dir.name, diff_dir.name)
                    records.append(TaskRecord(
                        subject=subj_dir.name,
                        state=state_dir.name,
                        difficulty=diff_dir.name,
                        video_path=vid,
                        errors_path=err_file,
                    ))
        return records

    def _find_video(self, subject: str, state: str, difficulty: str) -> Path:
        """尝试多种目录约定定位视频文件。"""
        candidates = [
            self.data_root / subject / state / difficulty / "training_video.mp4",
            self.data_root / subject / difficulty / state / "training_video.mp4",
            self.data_root / subject / f"{state}_{difficulty}.mp4",
            self.data_root / subject / "training_video.mp4",
        ]
        for c in candidates:
            if c.exists():
                return c
        # 返回约定路径（即使不存在，方便后续判断）
        return candidates[0]

    # ---------- 切窗 ----------

    def build_windows_for_record(
        self,
        subject: str,
        state: str,
        difficulty: str,
        window_sec: int,
        stride_sec: int,
        fps: float = 30.0,
    ) -> list[dict]:
        """对单条 record 的误差序列切窗。

        Parameters
        ----------
        window_sec / stride_sec : 窗口长度与步长（秒）
            测试集应 stride=window 以避免重叠泄漏；
            训练集可 stride<window 做 50% 重叠增广。
        fps : 误差序列采样率（Hz）

        Returns
        -------
        list[dict] 每窗含 subject/state/difficulty/window_idx/data
        """
        err_file = self.dist_error_root / difficulty / subject / state / "all_errors.txt"
        if not err_file.exists():
            return []
        errors = np.loadtxt(err_file, dtype=float).ravel()
        win = int(window_sec * fps)
        stride = int(stride_sec * fps)
        if win <= 0 or stride <= 0:
            raise ValueError("window_sec 与 stride_sec 必须为正")
        if len(errors) < win:
            return []
        windows: list[dict] = []
        for i, s in enumerate(range(0, len(errors) - win + 1, stride)):
            windows.append({
                "subject": subject,
                "state": state,
                "difficulty": difficulty,
                "window_idx": s,
                "data": errors[s : s + win],
            })
        return windows

    # ---------- LOSO 划分 ----------

    def write_loso_folds(self, n_subjects: int = 20) -> None:
        """生成 LOSO 划分表 CSV：每折留 1 名受试者作测试。"""
        self.output_root.joinpath("splits").mkdir(parents=True, exist_ok=True)
        rows = [{"fold": i, "test_subject": f"{i + 1:02d}"} for i in range(n_subjects)]
        pd.DataFrame(rows).to_csv(
            self.output_root / "splits" / "loso_folds.csv", index=False
        )

    # ---------- 注视特征矩阵 ----------

    def build_gaze_feature_matrix(
        self,
        window_sec: int = 60,
        stride_sec: int = 60,
        fps: float = 30.0,
        thresholds: Optional[dict[str, float]] = None,
    ) -> pd.DataFrame:
        """对全部 record 切窗并提取注视特征，输出特征矩阵。

        按 spec §4.1 / §5.3：阈值阶段确定的三个固定阈值（overall/easy/hard）
        全部复用，对每个窗口同时计算三个阈值的 inside_ratio/longest_run，
        保证特征空间在所有样本上一致、无 NaN，且不重新选阈值。

        Parameters
        ----------
        thresholds : {"overall": T, "easy": T, "hard": T}；缺省用 spec 推荐值
        """
        if thresholds is None:
            thresholds = {"overall": 900.0, "easy": 975.0, "hard": 625.0}
        ext = GazeFeatureExtractor()
        # 三个阈值全部纳入特征空间，按固定顺序保证列稳定
        t_values = [thresholds["overall"], thresholds["easy"], thresholds["hard"]]
        rows: list[dict] = []
        for rec in self.discover_records():
            windows = self.build_windows_for_record(
                rec.subject, rec.state, rec.difficulty, window_sec, stride_sec, fps
            )
            for w in windows:
                # 对每个阈值各提一次特征，合并 dict
                # 基础统计量（mean/median/std/p95）阈值无关，会被同名键覆盖为同值
                feat: dict = {}
                for t in t_values:
                    feat.update(ext.extract_window_features(w["data"], t))
                feat.update({
                    "subject": rec.subject,
                    "label": 1 if rec.state == "alert" else 0,
                    "state": rec.state,
                    "difficulty": rec.difficulty,
                    "window_idx": w["window_idx"],
                })
                rows.append(feat)
        return pd.DataFrame(rows)

    # ---------- 多模态对齐特征矩阵 ----------

    @staticmethod
    def build_multimodal_feature_matrix(
        gaze_csv: str | Path,
        face_csv: str | Path,
    ) -> pd.DataFrame:
        """对齐注视与面部特征矩阵，输出融合特征矩阵。

        对应 spec v2.0 §5.2 时间对齐：注视序列与视频帧按 60s 窗口聚合，
        对齐误差 < 1 帧。由于 gaze/face 的 window_idx 是各自采样率下的帧索引，
        不能直接匹配，这里按 (subject, state, difficulty) 分组后按窗口序号对齐——
        两者窗口时长与步长相同（window_sec=60, stride_sec=30），
        故同 record 内窗口数量一致，按排序后的序号 1-1 对齐。

        Parameters
        ----------
        gaze_csv : dataset/features/gaze_features.csv
        face_csv : dataset/features/face_features.csv

        Returns
        -------
        DataFrame：gaze 特征列在前，face 特征列在后，meta 列（subject/label/
        state/difficulty/window_order）在最后。仅保留双方都有窗口的样本（inner join）。
        """
        gaze_df = pd.read_csv(gaze_csv, dtype={"subject": str})
        face_df = pd.read_csv(face_csv, dtype={"subject": str})

        meta = ["subject", "state", "difficulty"]
        gaze_feat_cols = [c for c in gaze_df.columns
                          if c not in meta + ["label", "window_idx"]]
        face_feat_cols = [c for c in face_df.columns
                          if c not in meta + ["label", "window_idx"]]

        # 为每行分配组内窗口序号（按 window_idx 排序后 0,1,2,...）
        def _assign_order(df: pd.DataFrame) -> pd.DataFrame:
            df = df.sort_values(meta + ["window_idx"]).reset_index(drop=True)
            df["window_order"] = (
                df.groupby(meta).cumcount()
            )
            return df

        g = _assign_order(gaze_df)
        f = _assign_order(face_df)

        # 列重命名：face 特征加 face_ 前缀避免冲突
        f_rename = f.rename(
            columns={c: f"face_{c}" for c in face_feat_cols}
        )
        # inner join
        merged = pd.merge(
            g[meta + ["label", "window_order"] + gaze_feat_cols],
            f_rename[meta + ["window_order"] + [f"face_{c}" for c in face_feat_cols]],
            on=meta + ["window_order"],
            how="inner",
        )
        # 列顺序：meta + gaze 特征 + face 特征
        col_order = (
            meta + ["label", "window_order"]
            + gaze_feat_cols
            + [f"face_{c}" for c in face_feat_cols]
        )
        return merged[col_order]
