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
    immune_state_p_conf1: float  # probability of a person getting a label 'diseased' for confounder value 1
    immune_state_p_conf2: float  # probability of a person getting a label 'diseased' for confounder value 2
    confounder_p_train: float  # probability that the confounder will have value 1 in the training simulation data
    confounder_p_test: float  # probability that the confounder will have value 1 in the test simulation data
    immune_state_implanting_rate: float  # percentage of receptors in AIRR that contain the immune signal of disease
    confounder_implanting_rate: float  # percentage of receptors in AIRR that contain confounder signal
    immune_signal: dict  # which motifs the immune signal contains used to construct immuneML Signal object
    confounder_signal: dict  # which motifs the confounder signal contains used to construct immuneML Signal object
    sequence_count: int  # how many receptor sequences to create in one AIRR
    train_example_count: int  # how many AIRRs to make in the training dataset
    test_example_count: int  # how many AIRRs to make in the test dataset


@dataclass
class Experiment1:
    """
    Experiment1 class encapsulates experimental runs illustrating the influence of changing confounder
    distribution on prediction performance of ML models. It relies on Exp1Config to include all parameters
    needed to simulate the data from the causal graph and on immuneML package to perform ML training and assessment.

    The whole analysis with simulation and ML training and assessment is repeated multiple times
    (user-specified parameter) to obtain more robust estimates of the performance.
    """
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
