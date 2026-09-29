"""
阶段 0 验收测试：确认工程骨架可用。

这些测试不依赖任何外部服务（Qwen / 麦克风），
只验证目录结构、依赖、配置加载是否正确。
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_project_structure_exists():
    """工程目录结构完整。"""
    for d in ["app", "app/api", "app/core", "app/services",
              "app/storage", "app/web", "docs", "tests"]:
        assert (PROJECT_ROOT / d).is_dir(), f"缺少目录: {d}"


def test_prototype_is_archived():
    """原型已冻结归档，不再位于工程根目录。"""
    assert (PROJECT_ROOT / "prototype").is_dir()
    assert not (PROJECT_ROOT / "server.py").exists(), \
        "原型 server.py 不应留在根目录"


def test_core_dependencies_importable():
    """核心依赖可导入。"""
    import fastapi          # noqa: F401
    import numpy            # noqa: F401
    import uvicorn          # noqa: F401
    import websockets       # noqa: F401


def test_voiceprint_model_available():
    """声纹模型文件存在（原型资产，将搬入新工程）。"""
    model = PROJECT_ROOT / "prototype" / "models" / "campplus.onnx"
    assert model.is_file(), "声纹模型缺失"
    assert model.stat().st_size > 1_000_000, "声纹模型文件异常偏小"


def test_requirements_pinned():
    """requirements.txt 存在且全部为固定版本。"""
    req = PROJECT_ROOT / "requirements.txt"
    assert req.is_file()
    lines = [ln.strip() for ln in req.read_text(encoding="utf-8").splitlines()]
    pkgs = [ln for ln in lines if ln and not ln.startswith("#")]
    assert pkgs, "requirements.txt 为空"
    for ln in pkgs:
        assert "==" in ln, f"未固定版本: {ln}"
