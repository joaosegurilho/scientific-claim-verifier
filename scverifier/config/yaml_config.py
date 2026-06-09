from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class ExperimentSection:
    name: str
    description: Optional[str] = None


@dataclass
class ModelConfig:
    pipeline_model: Optional[str] = None
    agent_model: Optional[str] = None
    embedding_model: Optional[str] = None
    llm_temperature: Optional[float] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


VALID_DATASETS = {"scifact", "coverbench", "healthver", "msvec"}
VALID_METHODS = {"agent", "agentless", "agent_with_search", "agentless_with_search"}
VALID_SPLITS = {"train", "dev", "test", "all", None}


@dataclass
class BenchmarkRunConfig:
    dataset: str
    method: str
    split: Optional[str] = None
    max_claims: Optional[int] = None
    max_papers: Optional[int] = None

    def is_valid(self) -> bool:
        if self.dataset not in VALID_DATASETS:
            return False
        if self.method not in VALID_METHODS:
            return False
        if self.split not in VALID_SPLITS:
            return False
        return True


@dataclass
class CostTracking:
    enabled: bool = False
    budget_usd: Optional[float] = None


@dataclass
class ExperimentConfig:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    model: ModelConfig = field(default_factory=ModelConfig)
    kb: Any = None
    features: List[str] = field(default_factory=list)
    benchmark: List[BenchmarkRunConfig] = field(default_factory=list)
    cost_tracking: CostTracking = field(default_factory=CostTracking)


def build_experiment_config(data: Dict) -> ExperimentConfig:
    exp = data.get("experiment", {}) or {}
    experiment_section = ExperimentSection(
        name=exp.get("name", ""),
        description=exp.get("description"),
    )

    model = data.get("model", {}) or {}
    model_config = ModelConfig(
        pipeline_model=model.get("pipeline_model"),
        agent_model=model.get("agent_model"),
        embedding_model=model.get("embedding_model"),
        llm_temperature=model.get("llm_temperature"),
        chunk_size=model.get("chunk_size"),
        chunk_overlap=model.get("chunk_overlap"),
    )

    kb = data.get("kb", {}).get("db_name")
    kb_path = Path("data") / f"{kb}.db" if kb else None

    features = data.get("features", []) or []

    benchmark_items = []
    for item in data.get("benchmark") or []:
        benchmark_items.append(
            BenchmarkRunConfig(
                dataset=item.get("dataset", ""),
                split=None if item.get("split") == "all" else item.get("split"),
                max_claims=item.get("max_claims"),
                method=item.get("method"),
                max_papers=item.get("max_papers"),
            )
        )

    if benchmark_items and not all(item.is_valid() for item in benchmark_items):
        raise ValueError(
            "Invalid benchmark dataset or method in config. Must be one of: "
            f"dataset in {sorted(VALID_DATASETS)} and method in {sorted(VALID_METHODS)}."
        )

    cost = data.get("cost_tracking", {}) or {}
    cost_tracking = CostTracking(
        enabled=bool(cost.get("enabled", False)),
        budget_usd=cost.get("budget_usd"),
    )

    return ExperimentConfig(
        experiment=experiment_section,
        model=model_config,
        kb=kb_path,
        features=features,
        benchmark=benchmark_items,
        cost_tracking=cost_tracking,
    )
