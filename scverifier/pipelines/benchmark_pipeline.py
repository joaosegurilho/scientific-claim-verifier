from datetime import datetime
from pathlib import Path

import yaml

from scverifier.config.override import apply_experiment_config
from scverifier.config.settings import Config
from scverifier.config.yaml_config import ExperimentConfig, build_experiment_config
from scverifier.core.benchmarking import BENCHMARK_MAP, METHOD_MAP, get_benchmark
from scverifier.core.benchmarking.run_benchmark import BenchmarkRunner


class BenchmarkPipeline:
    def __init__(self, config_path):
        self.config_path: Path = config_path
        self.parsed_config: ExperimentConfig = build_experiment_config(yaml.safe_load(self.config_path.open()))
        self._has_valid_features()()

        self.expr_path: Path = Path("experiments")
        self.expr_path.mkdir(parents=True, exist_ok=True)
        self.counter_file: Path = self.expr_path / ".counter"
        if not self.counter_file.exists():
            self.counter_file.write_text("0")

    def __call__(self):

        apply_experiment_config(self.parsed_config)

        final_config_path = self._get_outputdir_path() / "config.yaml"
        final_config_path.write_text(yaml.dump(yaml.safe_load(self.config_path.open())))

        final_settings_snapshot_path = self._get_outputdir_path() / "settings_snapshot.yaml"
        final_settings_snapshot_path.write_text(
            yaml.dump({k: getattr(Config, k) for k in dir(Config) if k.isupper() and not k.startswith("_")})
        )

        for bm_item in self.parsed_config.benchmark:
            print(f"\nBenchmark: {bm_item.dataset}/{bm_item.method} (split={bm_item.split or 'all'})")
            benchmark = get_benchmark(bm_item.dataset, bm_item.method, bm_item.split)
            runner = BenchmarkRunner(
                benchmark=benchmark,
                results_dir=self._get_outputdir_path(),
                method=METHOD_MAP[bm_item.method],
                run_dir=self._get_outputdir_path() / f"{bm_item.dataset}_{bm_item.method}",
            )
            runner.run(max_papers=bm_item.max_papers or 10)

        self._set_expr_cnt()

    def dry_run(self):
        print("Dry run — experiment plan:")
        print(f"  Name:        {self.parsed_config.experiment.name or 'Unnamed'}")
        print(f"  Description: {self.parsed_config.experiment.description or '-'}")
        print(f"  Agent model: {self.parsed_config.model.agent_model or '(default)'}")
        print(f"  Embedding:   {self.parsed_config.model.embedding_model or '(default)'}")
        print(f"  KB:          {self.parsed_config.kb or '(default)'}")
        print(f"  Features:    {self.parsed_config.features or 'none'}")
        print(f"  Output:      {self._get_outputdir_path()}")
        print(f"  Benchmarks:  {len(self.parsed_config.benchmark)}")

        for item in self.parsed_config.benchmark:
            print(f"\t- {item.dataset}/{item.method} split={item.split or 'all'}")
            self._check_bechmark_dataset_ok(item)

    def _get_expr_cnt(self):
        return int(self.counter_file.read_text().strip()) + 1

    def _set_expr_cnt(self):
        self.counter_file.write_text(str(self._get_expr_cnt()))

    def _get_outputdir_path(self) -> Path:
        outdir = self.expr_path / Path(
            f"/results/exp_{self._get_expr_cnt():03d}_{self.parsed_config.experiment.name}_{datetime.now().strftime('%Y%m%d')}"
        )
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    def _has_valid_features(self):
        unknown = set(self.parsed_config.features) - Config.KNOWN_FEAUTURES
        if unknown:
            raise ValueError(f"Unknown features: {unknown}. --> Known: {Config.KNOWN_FEAUTURES}")

    def _check_bechmark_dataset_ok(self, benchmark_item):
        benchmark_cls = BENCHMARK_MAP[benchmark_item.dataset]
        vm = METHOD_MAP[benchmark_item.method]
        try:
            if benchmark_item.dataset == "scifact":
                bm = benchmark_cls(split=benchmark_item.split, verification_method=vm)
            else:
                bm = benchmark_cls(verification_method=vm)

            bm.load(max_items=1)
            print("      ✓ data available")

        except FileNotFoundError as e:
            print(f"      ✗ data not found: {e}")

        except Exception as e:
            print(f"      ! load error: {e}")
