import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
from immuneML.simulation.implants.Signal import Signal
from immuneML.util.PathBuilder import PathBuilder
from dagsim.base import Graph, Node, Selection

from causal_airr_scripts.dataset_util import make_olga_repertoire, make_dataset, make_AIRR_dataset
from causal_airr_scripts.implanting import make_repertoire_with_signal, make_repertoire_without_signal


def get_immune_state(confounder: str, p_conf1: float, p_conf2: float) -> bool:
    state = bool(np.random.binomial(n=1, p=p_conf2) if confounder == "C1" else np.random.binomial(n=1, p=p_conf1))
    return state


def get_confounder(p) -> str:
    c = "C1" if np.random.binomial(n=1, p=p) else "C2"
    return c


def get_repertoire(immune_state: bool, confounder: str, path: Path, naive_rep_path: Path, sequence_count: int,
                   immune_signal: Signal, confounder_signal: Signal, immune_state_implanting_rate: float,
                   confounder_implanting_rate: float) -> Path:
    PathBuilder.build(path)
    PathBuilder.build(naive_rep_path)

    # make olga repertoire using the default model
    naive_repertoire = make_olga_repertoire(path=naive_rep_path, sequence_count=sequence_count)

    # implant a confounder signal in the repertoire based on the confounder value
    if confounder == "C1":
        repertoire = make_repertoire_with_signal(naive_repertoire, signal=confounder_signal,
                                                 result_path=path / 'immuneML_with_confounder/',
                                                 repertoire_implanting_rate=confounder_implanting_rate)
    else:
        repertoire = make_repertoire_without_signal(repertoire=naive_repertoire, signal_name=confounder_signal.id,
                                                    result_path=path / "immuneML_with_confounder/")

    # implant a signal in the repertoire based on the immune state and confounder
    if immune_state:
        repertoire = make_repertoire_with_signal(repertoire=repertoire, signal=immune_signal,
                                                 result_path=path / "immuneML_with_signal/",
                                                 repertoire_implanting_rate=immune_state_implanting_rate)
    else:
        repertoire = make_repertoire_without_signal(repertoire=repertoire, signal_name=immune_signal.id,
                                                    result_path=path / "immuneML_with_signal/")

    return repertoire.data_filename


def get_selection(immune_state: bool, threshold: int, immune_state_counter: dict) -> bool:
    keep = True

    if immune_state_counter["true"] > immune_state_counter["false"] and immune_state_counter["true"] - immune_state_counter["false"] > threshold:
        if immune_state:
            keep = False
        else:
            immune_state_counter["false"] += 1
    elif immune_state_counter["true"] < immune_state_counter["false"] and immune_state_counter["false"] - immune_state_counter["true"] > threshold:
        if immune_state:
            immune_state_counter["true"] += 1
        else:
            keep = False
    elif immune_state:
        immune_state_counter["true"] += 1
    else:
        immune_state_counter["false"] += 1

    return keep


def simulate_dataset(train_example_count: int, test_example_count, data_path: Path, confounder_p_train, confounder_p_test: float,
                     immune_signal: Signal, experiment_name: str, immune_state_p_conf1, immune_state_p_conf2, confounder_signal,
                     immune_state_implanting_rate, confounder_implanting_rate, sequence_count):
    # make train dataset

    train_graph = make_graph(confounder_p=confounder_p_train, immune_state_p_conf1=immune_state_p_conf1,
                             immune_state_p_conf2=immune_state_p_conf2, data_path=data_path, sequence_count=sequence_count, train=True,
                             immune_signal=immune_signal, confounder_signal=confounder_signal, experiment_name=experiment_name,
                             immune_state_implanting_rate=immune_state_implanting_rate, confounder_implanting_rate=confounder_implanting_rate)

    study_cohort_data = train_graph.simulate(num_samples=train_example_count, csv_name=str(data_path / "train/study_cohort"), selection=True)

    # make an AIRR dataset from the generated repertoires to be used for training

    train_dataset = make_dataset(repertoire_paths=study_cohort_data["repertoire"], path=data_path / 'train',
                                 dataset_name=f"experiment{experiment_name}_train", signal_names=[immune_signal.id, 'confounder'])

    # # # make a test dataset

    test_graph = make_graph(confounder_p_test, immune_state_p_conf1, immune_state_p_conf2, data_path, sequence_count, immune_signal,
                            confounder_signal, immune_state_implanting_rate, confounder_implanting_rate, experiment_name, train=False)

    test_data = test_graph.simulate(num_samples=test_example_count, csv_name=str(data_path / "test/test_cohort"), selection=True)

    test_dataset = make_dataset(repertoire_paths=test_data["repertoire"], path=data_path / 'test',
                                dataset_name=f"experiment{experiment_name}_test", signal_names=[immune_signal.id, 'confounder'])

    # merge datasets (but the distinction between train and test will be kept)

    make_AIRR_dataset(train_dataset, test_dataset, data_path / 'full_dataset')

    # remove tmp files

    shutil.move(str(data_path / f"train/experiment{experiment_name}_train_metadata.csv"), data_path)
    shutil.move(str(data_path / f"test/experiment{experiment_name}_test_metadata.csv"), data_path)

    shutil.rmtree(data_path / 'train')
    shutil.rmtree(data_path / 'test')
    shutil.rmtree(data_path / 'naive')


def make_confounder_node(confounder_p):
    return Node(name="confounder", function=get_confounder, kwargs={"p": confounder_p})


def make_immune_state_node(confounder_node, immune_state_p_conf1, immune_state_p_conf2):
    return Node(name="immune_state", function=get_immune_state, kwargs={"confounder": confounder_node, "p_conf1": immune_state_p_conf1,
                                                                        "p_conf2": immune_state_p_conf2})


def make_selection_node(immune_state_node):
    immune_state_counter = {"true": 0, "false": 0}
    return Selection(name="S", function=get_selection, kwargs={"immune_state": immune_state_node, 'threshold': 4,
                                                               "immune_state_counter": immune_state_counter})


def make_repertoire_node(immune_state_node, confounder_node, data_path, train, immune_state_signal, sequence_count, confounder_signal,
                         immune_state_implanting_rate, confounder_implanting_rate):
    return Node(name="repertoire", function=get_repertoire,
                kwargs={"immune_state": immune_state_node, "confounder": confounder_node,
                        "path": data_path / "train" if train else data_path / "test", "naive_rep_path": data_path / "naive",
                        "sequence_count": sequence_count, "immune_signal": immune_state_signal,
                        "confounder_signal": confounder_signal,
                        'immune_state_implanting_rate': immune_state_implanting_rate,
                        'confounder_implanting_rate': confounder_implanting_rate})


def make_graph(confounder_p: float, immune_state_p_conf1: float, immune_state_p_conf2: float, data_path: Path, sequence_count: int,
               immune_signal: Signal, confounder_signal: Signal, immune_state_implanting_rate: float, confounder_implanting_rate: float,
               experiment_name: str, train: bool):
    confounder_node = make_confounder_node(confounder_p)

    immune_state_node = make_immune_state_node(confounder_node, immune_state_p_conf1, immune_state_p_conf2)

    repertoire_node = make_repertoire_node(immune_state_node, confounder_node, data_path, train, immune_signal, sequence_count,
                                           confounder_signal, immune_state_implanting_rate, confounder_implanting_rate)

    selection_node = make_selection_node(immune_state_node)

    # make a causal graph using DagSim

    graph = Graph(name=f"graph_experiment_{experiment_name}_{'train' if train else 'test'}",
                  list_nodes=[confounder_node, immune_state_node, repertoire_node, selection_node])

    return graph
