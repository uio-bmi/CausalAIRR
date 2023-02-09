from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor as Pool


import yaml
from immuneML.app.ImmuneMLApp import ImmuneMLApp

from causal_airr_scripts.dataset_util import setup_path
from causal_airr_scripts.implanting import make_signal
from causal_airr_scripts.ml_util import define_specs
from causal_airr_scripts.experiment2.exp2_util import simulate_dataset


@dataclass
class Exp2Config:
    p_immune_state: float
    p_hospital: float
    immune_state_implanting_rate: float
    protocol_implanting_rate: float
    immune_signal: dict
    protocol_signal_name: str
    sequence_count: int
    train_example_count: int
    test_example_count: int


@dataclass
class Experiment2:
    name: str
    result_path: Path
    config: Exp2Config
    repetitions: int
    num_processes: int

    def run(self) -> list:
        with Pool(self.num_processes) as pool:
            results = pool.map(self._run_repetition, list(range(1, self.repetitions + 1)))

        return list(results)

    def _run_repetition(self, repetition: int = 1):
        path = setup_path(self.result_path / f'repetition_{repetition}')
        data_path = setup_path(path / 'data')
        ml_path = setup_path(path / 'ml_result')

        simulate_dataset(train_example_count=self.config.train_example_count, test_example_count=self.config.test_example_count,
                         data_path=data_path, sequence_count=self.config.sequence_count, experiment_name=self.name, p_hospital=self.config.p_hospital,
                         p_immune_state=self.config.p_immune_state, immune_signal=make_signal(**self.config.immune_signal),
                         protocol_signal_name=self.config.protocol_signal_name, immune_state_implanting_rate=self.config.immune_state_implanting_rate,
                         protocol_implanting_rate=self.config.protocol_implanting_rate)

        specs_path = self._write_ml_specs(data_path, ml_path)

        app = ImmuneMLApp(specification_path=specs_path, result_path=ml_path / "result/")
        return app.run()

    def _write_ml_specs(self, data_path: Path, ml_path: Path) -> Path:
        specs = define_specs(data_path, experiment_name=self.name)

        specification_path = ml_path / "specs.yaml"

        with open(specification_path, "w") as file:
            yaml.dump(specs, file)

        return specification_path
