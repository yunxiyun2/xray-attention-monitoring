"""经典 ML 模型包装（LR / SVM / RF）。

对应 spec v2.0 §6.1 经典 ML 基线。
2026 Sensors 实证 LOSO 下 LR 反超 LSTM，故作为小样本下界与过拟合检测。
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def make_classical(name: str, random_state: int = 42):
    """工厂函数：按名称返回经典 ML 模型实例。

    所有模型统一暴露 fit / predict / predict_proba 接口。
    """
    if name == "logreg":
        return LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            random_state=random_state,
        )
    if name == "svm":
        return SVC(
            C=1.0, kernel="rbf", probability=True,
            class_weight="balanced", random_state=random_state,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=random_state,
        )
    raise ValueError(f"未知经典模型: {name}（可选: logreg / svm / rf）")
