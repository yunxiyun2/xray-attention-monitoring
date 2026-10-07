"""pytest 全局配置：确保 src 在 Python 路径中。"""
import sys
from pathlib import Path

src = Path(__file__).parent / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
