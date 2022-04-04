from pathlib import Path
from typing import List

import numpy as np
from immuneML.simulation.implants.Signal import Signal
from immuneML.util.PathBuilder import PathBuilder
from dagsim.base import Graph, Node

from util.dataset_util import make_olga_repertoire, make_dataset, make_AIRR_dataset
from util.implanting import make_repertoire_with_signal, make_repertoire_without_signal


def get_immune_state(confounder: str, p_conf1: float, p_conf2: float) -> bool:
    return bool(np.random.binomial(n=1, p=p_conf2) if confounder == "C1" else np.random.binomial(n=1, p=p_conf1))


def get_confounder(p) -> str:
    return "C1" if np.random.binomial(n=1, p=p) else "C2"


def get_repertoire(immune_state: bool, confounder: str, path: Path, naive_rep_path: Path, sequence_count: int,
                   immune_state_signals: List[Signal], confounder_signal: Signal, immune_state_implanting_rate: float,
                   confounder_implanting_rate: float) -> Path:
    PathBuilder.build(path)
    PathBuilder.build(naive_rep_path)

    # make olga repertoire using the default model
    naive_repertoire = make_olga_repertoire(path=naive_rep_path, sequence_count=sequence_count)

    # implant a signal in the repertoire based on the confounder value
    if confounder == "C1":
        repertoire = make_repertoire_with_signal(naive_repertoire, signal=confounder_signal,
                                                 result_path=path / 'immuneML_with_confounder/',
                                                 repertoire_implanting_rate=confounder_implanting_rate)
    else:
        repertoire = make_repertoire_without_signal(repertoire=naive_repertoire, signal_name=confounder_signal.id,
                                                    result_path=path / "immuneML_with_confounder/")

    # implant a signal in the repertoire based on the immune state and confounder
    if immune_state:
        signal = immune_state_signals[0] if confounder == "C1" else immune_state_signals[1]
        repertoire = make_repertoire_with_signal(repertoire=repertoire, signal=signal,
                                                 result_path=path / "immuneML_with_signal/",
                                                 repertoire_implanting_rate=immune_state_implanting_rate)
    else:
        repertoire = make_repertoire_without_signal(repertoire=repertoire, signal_name=immune_state_signals[0].id,
                                                    result_path=path / "immuneML_with_signal/")

    return repertoire.data_filename


def simulate_dataset(train_example_count: int, test_example_count, data_path: Path, graph: Graph, confounder_p_test,
                     sequence_count: int, immune_state_signals: list, confounder_signal, immune_state_implanting_rate: float,
                     confounder_implanting_rate: float, experiment_name: str):

    confounder_node = [node for node in graph.nodes if node.name == "confounder"][0]
    immune_state_node = [node for node in graph.nodes if node.name == 'immune_state'][0]

    # make train dataset
    study_cohort_data = graph.simulate(num_samples=train_example_count, csv_name=str(data_path / "train/study_cohort"))

    # make an AIRR dataset from the generated repertoires to be used for training

    train_dataset = make_dataset(repertoire_paths=study_cohort_data["repertoire"], path=data_path / 'train',
                                 dataset_name=f"experiment{experiment_name}_train",
                                 signal_names=[immune_state_signals[0].id, confounder_node.name])

    # make a test dataset
    confounder_node = Node(name="confounder", function=get_confounder, kwargs={"p": confounder_p_test})

    repertoire_node = Node(name="repertoire", function=get_repertoire,
                           kwargs={"immune_state": immune_state_node, "confounder": confounder_node, "path": data_path / "test",
                                   "naive_rep_path": data_path / "naive", "sequence_count": sequence_count,
                                   "immune_state_signals": immune_state_signals, "confounder_signal": confounder_signal,
                                   'immune_state_implanting_rate': immune_state_implanting_rate,
                                   'confounder_implanting_rate': confounder_implanting_rate})

    graph = Graph(name=f"graph_experiment_{experiment_name}", list_nodes=[confounder_node, immune_state_node, repertoire_node])

    test_data = graph.simulate(num_samples=test_example_count, csv_name=str(data_path / "test/test_cohort"))

    test_dataset = make_dataset(repertoire_paths=test_data["repertoire"], path=data_path / 'test',
                                dataset_name=f"experiment{experiment_name}_test",
                                signal_names=[immune_state_signals[0].id, confounder_node.name])

    # merge datasets (but the distinction between train and test will be kept)

    dataset = make_AIRR_dataset(train_dataset, test_dataset, data_path / 'full_dataset')


def define_specs(data_path: Path, experiment_name: str) -> dict:
    return {
        "definitions": {
            "datasets": {
                "dataset1": {
                    "format": 'AIRR',
                    "params": {
                        "path": str(data_path / 'full_dataset'),
                        "metadata_file": str(data_path / 'full_dataset/metadata.csv')
                    }
                }
            },
            "encodings": {
                "kmer_frequency": {
                    "KmerFrequency": {"k": 3}
                }
            },
            "ml_methods": {
                "logistic_regression": {
                    "LogisticRegression": {
                        "penalty": "l1",
                        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        "show_warnings": False
                    },
                    "model_selection_cv": True,
                    "model_selection_n_folds": 5
                }
            },
            "reports": {
                "coefficients": {
                    "Coefficients": {  # show top 25 logistic regression coefficients and what k-mers they correspond to
                        "coefs_to_plot": ['n_largest'],
                        "n_largest": [25]
                    }
                },
                "feature_comparison": {
                    "FeatureComparison": {
                        "comparison_label": "immune_state",
                        "color_grouping_label": "confounder",
                        "show_error_bar": False,
                        "keep_fraction": 0.1,
                        "log_scale": True
                    }
                }
            }
        },
        "instructions": {
            'train_ml': {
                "type": "TrainMLModel",
                "assessment": {  # ensure here that train and test dataset are fixed, as per simulation
                    "split_strategy": "manual",
                    "split_count": 1,
                    "manual_config": {
                        "train_metadata_path": str(data_path / f"train/experiment{experiment_name}_train_metadata.csv"),
                        "test_metadata_path": str(data_path / f"test/experiment{experiment_name}_test_metadata.csv")
                    },
                    "reports": {
                        "models": ["coefficients"],
                        "encoding": ["feature_comparison"]
                    }
                },
                "selection": {
                    "split_strategy": "k_fold",
                    "split_count": 5,
                    "reports": {
                        "models": ["coefficients"],
                        "encoding": ["feature_comparison"]
                    }
                },
                "settings": [
                    {"encoding": "kmer_frequency", "ml_method": "logistic_regression"}
                ],
                "dataset": "dataset1",
                "refit_optimal_model": False,
                "labels": ["immune_state"],
                "optimization_metric": "balanced_accuracy",
                "metrics": ['log_loss', 'auc']
            }
        }
    }


def make_graph(confounder_p_train: float, immune_state_p_conf1: float, immune_state_p_conf2: float, data_path: Path, sequence_count: int,
               immune_state_signals: list, confounder_signal: Signal, immune_state_implanting_rate: float, confounder_implanting_rate: float,
               experiment_name: str):

    confounder_node = Node(name="confounder", function=get_confounder, kwargs={"p": confounder_p_train})

    immune_state_node = Node(name="immune_state", function=get_immune_state,
                             kwargs={"confounder": confounder_node, "p_conf1": immune_state_p_conf1,
                                     "p_conf2": immune_state_p_conf2})

    repertoire_node = Node(name="repertoire", function=get_repertoire,
                           kwargs={"immune_state": immune_state_node, "confounder": confounder_node,
                                   "path": data_path / "train", "naive_rep_path": data_path / "naive",
                                   "sequence_count": sequence_count,
                                   "immune_state_signals": immune_state_signals,
                                   "confounder_signal": confounder_signal,
                                   'immune_state_implanting_rate': immune_state_implanting_rate,
                                   'confounder_implanting_rate': confounder_implanting_rate})

    # make a causal graph using DagSim

    graph = Graph(name=f"graph_experiment_{experiment_name}", list_nodes=[confounder_node, immune_state_node, repertoire_node])

    return graph
