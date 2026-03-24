from tackle_vision_ml.config import PipelineConfig


def test_default_config_constructs():
    cfg = PipelineConfig()
    assert cfg.device in {"cpu", "cuda"}
    assert cfg.frame_stride >= 1

