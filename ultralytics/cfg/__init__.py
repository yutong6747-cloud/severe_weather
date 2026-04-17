# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_PATH, yaml_load

MODES = {"train", "val", "predict", "export", "track", "benchmark"}
TASKS = {"detect", "segment", "classify", "pose", "obb"}
TASK2DATA = {
    "detect": "coco8.yaml",
    "segment": "coco8-seg.yaml",
    "classify": "imagenet10",
    "pose": "coco8-pose.yaml",
    "obb": "dota8.yaml",
}
TASK2MODEL = {
    "detect": "yolo11n.pt",
    "segment": "yolo11n-seg.pt",
    "classify": "yolo11n-cls.pt",
    "pose": "yolo11n-pose.pt",
    "obb": "yolo11n-obb.pt",
}
TASK2METRIC = {
    "detect": "metrics/mAP50-95(B)",
    "segment": "metrics/mAP50-95(M)",
    "classify": "metrics/accuracy_top1",
    "pose": "metrics/mAP50-95(P)",
    "obb": "metrics/mAP50-95(B)",
}
MODELS = {TASK2MODEL[task] for task in TASKS}
DEFAULT_CFG = Path(DEFAULT_CFG_PATH)


def cfg2dict(cfg: Any) -> Dict[str, Any]:
    """Convert a YAML path, dict, or namespace into a dictionary."""
    if isinstance(cfg, (str, Path)):
        return yaml_load(cfg)
    if isinstance(cfg, SimpleNamespace):
        return vars(cfg)
    return dict(cfg)


def get_cfg(cfg: Any = DEFAULT_CFG_DICT, overrides: Dict[str, Any] | None = None) -> SimpleNamespace:
    """Return a simple namespace with merged configuration values."""
    data = cfg2dict(cfg)
    if overrides:
        data.update(cfg2dict(overrides))
    return SimpleNamespace(**data)


def get_save_dir(args: Any) -> Path:
    """Create and return the default save directory."""
    project = getattr(args, "project", None) or "runs"
    name = getattr(args, "name", None) or "exp"
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


__all__ = (
    "DEFAULT_CFG",
    "MODELS",
    "MODES",
    "TASK2DATA",
    "TASK2METRIC",
    "TASK2MODEL",
    "TASKS",
    "cfg2dict",
    "get_cfg",
    "get_save_dir",
)
