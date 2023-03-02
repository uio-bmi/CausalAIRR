import shutil
from pathlib import Path

import numpy as np
import yaml
from immuneML.simulation.implants.Signal import Signal
from immuneML.util.PathBuilder import PathBuilder
from dagsim.base import Node, Selection, Graph

from causal_airr_scripts.dataset_util import make_olga_repertoire, make_dataset, make_AIRR_dataset
from causal_airr_scripts.implanting import make_repertoire_with_signal, make_repertoire_without_signal, make_exp_protocol_signal


def get_immune_state(p: float) -> bool:
    return bool(np.random.binomial(n=1, p=p))


def get_hospital(p) -> str:
    return "hospital1" if np.random.binomial(n=1, p=p) else "hospital2"


def get_exp_protocol(hospital: str) -> int:
    protocol_id = 1 if hospital == "hospital1" else 2
    return protocol_id


def get_selection(hospital: str, immune_state: bool) -> bool:
    if hospital == "hospital1":
        if immune_state:
            return bool(np.random.binomial(1, 0.8))
        else:
            return bool(np.random.binomial(1, 0.2))
    if hospital == "hospital2":
        if immune_state:
            return bool(np.random.binomial(1, 0.2))
        else:
            return bool(np.random.binomial(1, 0.8))


def get_repertoire(immune_state: bool, experimental_protocol_id: int, path: Path, sequence_count: int, protocol_name: str,
                   immune_signal: Signal, immune_state_implanting_rate: float, protocol_implanting_rate: float) -> str:
    PathBuilder.build(path)

    repertoire = make_olga_repertoire(path=path, sequence_count=sequence_count)

    # implant a signal in the repertoire based on the immune state
    if immune_state:
        repertoire = make_repertoire_with_signal(repertoire=repertoire, signal=immune_signal, result_path=path / "immuneML_with_signal/",
                                                 repertoire_implanting_rate=immune_state_implanting_rate)
    else:
        repertoire = make_repertoire_without_signal(repertoire=repertoire, signal_name=immune_signal.id,
                                                    result_path=path / "immuneML_with_signal/")

    # simulate experimental protocol signal in the repertoire

    exp_protocol_signal = make_exp_protocol_signal(protocol_id=experimental_protocol_id, signal_name=protocol_name)

    repertoire = make_repertoire_with_signal(repertoire=repertoire, signal=exp_protocol_signal, result_path=path / "immuneML_with_protocol/",
                                             repertoire_implanting_rate=protocol_implanting_rate)

    update_protocol_metadata(repertoire.metadata_filename, experimental_protocol_id, protocol_name)

    return repertoire.data_filename


def update_protocol_metadata(metadata_filename: Path, experimental_protocol_id: int, exp_protocol_signal_name: str):
    with metadata_filename.open('r') as file:
        metadata = yaml.safe_load(file)

    metadata[exp_protocol_signal_name] = experimental_protocol_id

    with metadata_filename.open('w') as file:
        yaml.dump(metadata, file)


def make_graph(data_path, p_immune_state: float, p_hospital: float, immune_state_signal: Signal, sequence_count: int, experiment_name: str,
               immune_state_implanting_rate: float, protocol_implanting_rate: float, is_train: bool, protocol_name: str):
    immune_state_node = Node(name="immune_state", function=get_immune_state, kwargs={"p": p_immune_state})

    hospital_node = Node(name="hospital", function=get_hospital, kwargs={"p": p_hospital})

    experimental_protocol_node = Node(name=protocol_name, function=get_exp_protocol, kwargs={"hospital": hospital_node})

    repertoire_node = Node(name="repertoire", function=get_repertoire,
                           kwargs={"immune_state": immune_state_node, "experimental_protocol_id": experimental_protocol_node,
                                   "path": data_path / "train" if is_train else data_path / "test", "sequence_count": sequence_count,
                                   "immune_signal": immune_state_signal, 'immune_state_implanting_rate': immune_state_implanting_rate,
                                   "protocol_implanting_rate": protocol_implanting_rate, "protocol_name": protocol_name})

    selection_node = Selection(name="S", function=get_selection, kwargs={"hospital": hospital_node, "immune_state": immune_state_node})

    return Graph(name=experiment_name, list_nodes=[immune_state_node, hospital_node, experimental_protocol_node, repertoire_node, selection_node])


def simulate_dataset(data_path: Path, train_example_count: int, test_example_count: int, p_immune_state: float, p_hospital: float,
                     immune_signal: Signal, protocol_signal_name: str, sequence_count: int,
                     immune_state_implanting_rate: float, protocol_implanting_rate: float, experiment_name: str):

    train_graph = make_graph(data_path=data_path, p_immune_state=p_immune_state, p_hospital=p_hospital, immune_state_signal=immune_signal,
                             sequence_count=sequence_count, immune_state_implanting_rate=immune_state_implanting_rate,
                             protocol_implanting_rate=protocol_implanting_rate, experiment_name=experiment_name, is_train=True,
                             protocol_name=protocol_signal_name)

    signal_names = [immune_signal.id, protocol_signal_name]

    training_data = train_graph.simulate(num_samples=train_example_count, selection=True, csv_name=str(data_path / "train/study_cohort"))
    train_dataset = make_dataset(repertoire_paths=training_data["repertoire"], path=data_path / 'train',
                                 dataset_name=f"experiment{experiment_name}_train", signal_names=signal_names)

    test_graph = make_graph(data_path=data_path, p_immune_state=p_immune_state, p_hospital=p_hospital, immune_state_signal=immune_signal,
                            sequence_count=sequence_count, immune_state_implanting_rate=immune_state_implanting_rate,
                            protocol_implanting_rate=protocol_implanting_rate, experiment_name=experiment_name, is_train=False,
                            protocol_name=protocol_signal_name)

    test_data = test_graph.simulate(num_samples=test_example_count, csv_name=str(data_path / "test/test_cohort"), selection=False)
    test_dataset = make_dataset(repertoire_paths=test_data["repertoire"], path=data_path / 'test', dataset_name=f"experiment{experiment_name}_test",
                                signal_names=signal_names)

    # merge datasets (but the distinction between train and test will be kept in the ML analysis part)

    make_AIRR_dataset(train_dataset, test_dataset, data_path / 'full_dataset')

    # remove tmp files

    shutil.move(str(data_path / f"train/experiment{experiment_name}_train_metadata.csv"), data_path)
    shutil.move(str(data_path / f"test/experiment{experiment_name}_test_metadata.csv"), data_path)

    shutil.rmtree(data_path / 'train', ignore_errors=True)
    shutil.rmtree(data_path / 'test', ignore_errors=True)
    shutil.rmtree(data_path / 'naive', ignore_errors=True)
