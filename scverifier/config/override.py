from scverifier.config.settings import Config
from scverifier.config.yaml_config import ExperimentConfig


FIELD_MAP = {
    "pipeline_model": "LLM_MODEL",
    "agent_model": "AGENT_MODEL",
    "embedding_model": "EMBEDDING_MODEL",
    "llm_temperature": "LLM_TEMPERATURE",
    "chunk_size": "CHUNK_SIZE",
    "chunk_overlap": "CHUNK_OVERLAP",
}


def apply_experiment_config(exp_cfg: ExperimentConfig) -> None:
    for src_attr, cfg_attr in FIELD_MAP.items():
        value = getattr(exp_cfg.model, src_attr, None)
        if value is not None:
            setattr(Config, cfg_attr, value)

    if exp_cfg.features:
        unknown = set(exp_cfg.features) - Config.KNOWN_FEAUTURES
        if unknown:
            raise ValueError(
                f"Unknown features specified in config: {unknown}.\n" f"Known features are: {Config.KNOWN_FEAUTURES}"
            )
        Config.FEATURES = set(exp_cfg.features)

    if exp_cfg.kb is not None:
        Config.DB_NAME = str(exp_cfg.kb)
