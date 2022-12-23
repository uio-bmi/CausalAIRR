import argparse
import logging
from datetime import datetime
from itertools import product
from pathlib import Path

from causal_airr_scripts.dataset_util import setup_path
from causal_airr_scripts.experiment3.SimConfig import SimConfig, make_signal, ImplantingConfig, ImplantingGroup, ImplantingUnit
from causal_airr_scripts.experiment3.experiment3 import Experiment3
from causal_airr_scripts.util import write_config


def main(namespace):
    for p_noise, sequence_count in product([0.1, 0.499], [1000, 10000]):
        config = SimConfig(k=3, repetitions=10, p_noise=p_noise, olga_model_name='humanTRB',
                           signal=make_signal(motif_seeds=["YEQ", "PQH", "LFF"], seq_position_weights={108: 0.5, 109: 0.5},
                                              hamming_dist_weights={1: 0.5, 0: 0.5},
                                              position_weights={1: 1.}),
                           fdrs=[0.05], sequence_encoding='continuous_kmer',
                           implanting_config=ImplantingConfig(
                               control=ImplantingGroup(baseline=ImplantingUnit(0.5, []),
                                                       modified=ImplantingUnit(0.5, []), name='control', seq_count=sequence_count),
                               batch=ImplantingGroup(baseline=ImplantingUnit(0.9, []), name='batch', seq_count=sequence_count,
                                                     modified=ImplantingUnit(0.1, ["TRBV20", "TRBV5-1", "TRBV24", "TRBV27"]))))

        path = setup_path(namespace.result_path / f"experiment3_results/AIRR_classification_noise_{p_noise}_seqcount_{sequence_count}_{datetime.now()}")
        write_config(config, path)

        experiment = Experiment3(config, 0.01)
        experiment.run(path)


def main_test_run():
    config = SimConfig(k=3, repetitions=3, p_noise=0.1, olga_model_name='humanTRB',
                       signal=make_signal(motif_seeds=["YEQ", "PQH", "LFF"], seq_position_weights={108: 0.5, 109: 0.5},
                                          hamming_dist_weights={0: 0.5, 1: 0.5},
                                          position_weights={1: 1.}),
                       fdrs=[0.05], sequence_encoding='continuous_kmer',
                       implanting_config=ImplantingConfig(
                           control=ImplantingGroup(baseline=ImplantingUnit(0.5, []),
                                                   modified=ImplantingUnit(0.5, []), name='control', seq_count=100),
                           batch=ImplantingGroup(baseline=ImplantingUnit(0.9, []), name='batch', seq_count=100,
                                                 modified=ImplantingUnit(0.1, ["TRBV20", "TRBV5-1", "TRBV24", "TRBV27"]))))

    path = setup_path(f"./experiment3_results/AIRR_classification_{datetime.now()}")
    write_config(config, path)

    logging.basicConfig(level=logging.INFO)

    experiment = Experiment3(config, 0.0001)
    experiment.run(path)


def prepare_namespace():
    parser = argparse.ArgumentParser(description="CausalAIRR experiment 3")
    parser.add_argument("result_path", help="Output directory path.")

    namespace = parser.parse_args()
    namespace.result_path = Path(namespace.result_path)
    return namespace


if __name__ == "__main__":
    namespace = prepare_namespace()
    main(namespace)
