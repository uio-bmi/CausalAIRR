import argparse
import logging
from datetime import datetime
from itertools import product
from pathlib import Path

from causal_airr_scripts.dataset_util import setup_path
from causal_airr_scripts.experiment3.SimConfig import SimConfig, make_signal, ImplantingConfig, ImplantingSetting, ImplantingUnit, ImplantingGroup
from causal_airr_scripts.experiment3.experiment3 import Experiment3
from causal_airr_scripts.util import write_config


def main(namespace):

    test_set_percentage = 0.3

    for p_noise, sequence_count in product([0.1, 0.4, 0.45, 0.49], [500, 2500]):

        test_set_size = round(test_set_percentage * sequence_count)

        config = SimConfig(k=3, repetitions=5, p_noise=p_noise, olga_model_name='humanTRB', sequence_encoding='continuous_kmer',
                           signal=make_signal(motif_seeds=["YEQ", "PQH", "LFF"], seq_position_weights={108: 0.5, 109: 0.5},
                                              hamming_dist_weights={1: 0.8, 0: 0.2}, position_weights={1: 1.}),
                           implanting_config=ImplantingConfig(
                               control=ImplantingSetting(train=ImplantingGroup(baseline=ImplantingUnit(0.5, []), modified=ImplantingUnit(0.5, []),
                                                                               seq_count=sequence_count),
                                                         test=ImplantingGroup(baseline=ImplantingUnit(0.5, []), modified=ImplantingUnit(0.5, []),
                                                                              seq_count=test_set_size),
                                                         name='control'),
                               batch=ImplantingSetting(train=ImplantingGroup(baseline=ImplantingUnit(0.9, []), seq_count=sequence_count,
                                                                             modified=ImplantingUnit(0.1, ["TRBV20", "TRBV5-1", "TRBV24", "TRBV27"])),
                                                       test=ImplantingGroup(baseline=ImplantingUnit(0.5, []), seq_count=test_set_size,
                                                                            modified=ImplantingUnit(0.5, ["TRBV20", "TRBV5-1", "TRBV24", "TRBV27"])),
                                                       name='batch')))

        path = setup_path(namespace.result_path / f"experiment3_results/AIRR_classification_noise_{p_noise}_seqcount_{sequence_count}_{datetime.now()}")
        write_config(config, path)

        experiment = Experiment3(config, num_processes=namespace.num_processes)
        experiment.run(path)


def main_test_run():
    config = SimConfig(k=3, repetitions=2, p_noise=0.4, olga_model_name='humanTRB', sequence_encoding='continuous_kmer',
                       signal=make_signal(motif_seeds=["YEQ", "PQH", "LFF"], seq_position_weights={108: 0.33, 109: 0.34, 110: 0.33},
                                          hamming_dist_weights={0: 0.2, 1: 0.8}, position_weights={1: 1.}),
                       implanting_config=ImplantingConfig(
                               control=ImplantingSetting(train=ImplantingGroup(baseline=ImplantingUnit(0.5, []), modified=ImplantingUnit(0.5, []),
                                                                               seq_count=100),
                                                         test=ImplantingGroup(baseline=ImplantingUnit(0.5, []), modified=ImplantingUnit(0.5, []),
                                                                              seq_count=20),
                                                         name='control'),
                               batch=ImplantingSetting(train=ImplantingGroup(baseline=ImplantingUnit(0.9, []), seq_count=100,
                                                                             modified=ImplantingUnit(0.1, ["TRBV20", "TRBV5-1", "TRBV24", "TRBV27"])),
                                                       test=ImplantingGroup(baseline=ImplantingUnit(0.5, []), seq_count=20,
                                                                            modified=ImplantingUnit(0.5, ["TRBV20", "TRBV5-1", "TRBV24", "TRBV27"])),
                                                       name='batch')))

    path = setup_path(f"./experiment3_results/AIRR_classification_{datetime.now()}")
    write_config(config, path)

    logging.basicConfig(level=logging.INFO)

    experiment = Experiment3(config, num_processes=4)
    experiment.run(path)


def prepare_namespace():
    parser = argparse.ArgumentParser(description="CausalAIRR experiment 3")
    parser.add_argument("result_path", help="Output directory path.")
    parser.add_argument("num_processes", help="Number of processes to use for training logistic regression.", type=int)

    namespace = parser.parse_args()
    namespace.result_path = Path(namespace.result_path)
    return namespace


if __name__ == "__main__":
    namespace = prepare_namespace()
    main(namespace)
    # main_test_run()
