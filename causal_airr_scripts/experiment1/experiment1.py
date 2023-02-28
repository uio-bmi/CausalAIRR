from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor as Pool


import yaml
from immuneML.app.ImmuneMLApp import ImmuneMLApp

from causal_airr_scripts.dataset_util import setup_path
from causal_airr_scripts.implanting import make_signal
from causal_airr_scripts.ml_util import define_specs
from causal_airr_scripts.experiment1.exp1_util import simulate_dataset
from causal_airr_scripts.util import save_to_yaml


@dataclass
class Exp1Config:
    immune_state_p_conf1: float
    immune_state_p_conf2: float
    confounder_p_train: float
    confounder_p_test: float
    immune_state_implanting_rate: float
    confounder_implanting_rate: float
    immune_signal: dict
    confounder_signal: dict
    sequence_count: int
    train_example_count: int
    test_example_count: int


@dataclass
class Experiment1:
    name: str
    result_path: Path
    config: Exp1Config
    repetitions: int
    num_processes: int

    def run(self) -> list:

        self.write_config()

        with Pool(self.num_processes) as pool:
            results = pool.map(self._run_repetition, list(range(1, self.repetitions + 1)))

        return list(results)

    def write_config(self):
        save_to_yaml(vars(self.config), self.result_path / f'{self.name}_specs.yaml')

    def _run_repetition(self, repetition: int = 1):
        path = setup_path(self.result_path / f'repetition_{repetition}')
        data_path = setup_path(path / 'data')
        ml_path = setup_path(path / 'ml_result')

        simulate_dataset(train_example_count=self.config.train_example_count, test_example_count=self.config.test_example_count,
                         data_path=data_path, confounder_p_train=self.config.confounder_p_train, sequence_count=self.config.sequence_count,
                         confounder_p_test=self.config.confounder_p_test, immune_state_p_conf1=self.config.immune_state_p_conf1,
                         immune_state_p_conf2=self.config.immune_state_p_conf2, immune_state_implanting_rate=self.config.immune_state_implanting_rate,
                         confounder_implanting_rate=self.config.confounder_implanting_rate, experiment_name=self.name,
                         confounder_signal=make_signal(**self.config.confounder_signal), immune_signal=make_signal(**self.config.immune_signal))

        specs_path = self._write_ml_specs(data_path, ml_path)

        app = ImmuneMLApp(specification_path=specs_path, result_path=ml_path / "result/")
        return app.run()

    def _write_ml_specs(self, data_path: Path, ml_path: Path) -> Path:
        specs = define_specs(data_path, experiment_name=self.name)

        specification_path = ml_path / "specs.yaml"

        with open(specification_path, "w") as file:
            yaml.dump(specs, file)

        return specification_path
