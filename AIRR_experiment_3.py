import argparse
import logging
from datetime import datetime
from itertools import product
from multiprocessing import Pool
from pathlib import Path

from sklearn.linear_model import Ridge, LinearRegression

from causal_airr_scripts.dataset_util import setup_path
from causal_airr_scripts.experiment3.SimConfig import SimConfig, make_signal, ImplantingConfig, ImplantingSetting, ImplantingUnit, ImplantingGroup, \
    MotifProbGivenLabel, LabelProbSet
from causal_airr_scripts.experiment3.experiment3 import Experiment3
from causal_airr_scripts.util import write_config


def main(namespace):
    batch_effect_genes = ["TRBV20", "TRBV5-1", "TRBV24", "TRBV27"]

    proba_sets = [LabelProbSet(control=MotifProbGivenLabel(0.1, 0.8, implanting_prob=0.45),
                               batch0=MotifProbGivenLabel(0.1, 0.8, implanting_prob=0.17),
                               batch1=MotifProbGivenLabel(0.1, 0.8, implanting_prob=0.73), name='additional')]

    with Pool(processes=2) as pool:
        pool.starmap(run_one_config, [(proba_set, sequence_count, batch_effect_genes, namespace.result_path, namespace.num_processes)
                                      for index, (proba_set, sequence_count) in enumerate(product(proba_sets, [1000, 5000]))])


def run_one_config(proba_set, sequence_count, batch_effect_genes, result_path, num_processes):
    signal = make_signal(motif_seeds=["YEQ", "PQH", "LFF"], seq_position_weights={108: 0.5, 109: 0.5}, hamming_dist_weights={1: 0.8, 0: 0.2},
                         position_weights={1: 1.})

    config = SimConfig(k=3, repetitions=5, olga_model_name='humanTRB', sequence_encoding='continuous_kmer', signal=signal,
                       batch_corrections=[None, Ridge(alpha=1e8), Ridge(alpha=1e9)],
                       implanting_config=ImplantingConfig(
                           control=ImplantingSetting(
                               train=ImplantingGroup(baseline=ImplantingUnit(proba_set.control.implanting_prob, [], proba_set.control.motif_given_no_label_prob,
                                                                             proba_set.control.motif_given_label_prob),
                                                     modified=ImplantingUnit(proba_set.control.implanting_prob, [], proba_set.control.motif_given_no_label_prob,
                                                                             proba_set.control.motif_given_label_prob), seq_count=sequence_count),
                               test=ImplantingGroup(baseline=ImplantingUnit(proba_set.control.implanting_prob, [], proba_set.control.motif_given_no_label_prob,
                                                                            proba_set.control.motif_given_label_prob),
                                                    modified=ImplantingUnit(proba_set.control.implanting_prob, [], proba_set.control.motif_given_no_label_prob,
                                                                            proba_set.control.motif_given_label_prob), seq_count=sequence_count),
                               name='control'),
                           batch=ImplantingSetting(
                               train=ImplantingGroup(baseline=ImplantingUnit(proba_set.batch0.implanting_prob, [], proba_set.batch0.motif_given_no_label_prob,
                                                                             proba_set.batch0.motif_given_label_prob), seq_count=sequence_count,
                                                     modified=ImplantingUnit(proba_set.batch1.implanting_prob, batch_effect_genes, proba_set.batch1.motif_given_no_label_prob,
                                                                             proba_set.batch1.motif_given_label_prob)),
                               test=ImplantingGroup(baseline=ImplantingUnit(proba_set.control.implanting_prob, [], proba_set.control.motif_given_no_label_prob,
                                                                            proba_set.control.motif_given_label_prob),
                                                    seq_count=sequence_count,
                                                    modified=ImplantingUnit(proba_set.control.implanting_prob, batch_effect_genes,
                                                                            proba_set.control.motif_given_no_label_prob,
                                                                            proba_set.control.motif_given_label_prob)),
                               name='batch')))

    path = setup_path(result_path / f"experiment3_results/AIRR_classification_setup_{proba_set.name}_seqcount_{sequence_count}_{datetime.now()}")
    write_config(config, path)

    experiment = Experiment3(config, num_processes=num_processes)
    experiment.run(path)


def main_test_run():
    pos_given_no_motif_prob, pos_given_motif_prob = 40 / 85, 2 / 3
    seq_count = 500

    batch_effect_genes = ["TRBV20", "TRBV5-1", "TRBV24", "TRBV27"]

    config = SimConfig(k=3, repetitions=3, olga_model_name='humanTRB', sequence_encoding='continuous_kmer',
                       signal=make_signal(motif_seeds=["YEQ", "PQH", "LFF"], seq_position_weights={108: 0.5, 109: 0.5},
                                          hamming_dist_weights={0: 0.2, 1: 0.8}, position_weights={1: 1.}),
                       batch_corrections=[None, LinearRegression(), Ridge(alpha=1e3), Ridge(alpha=1e4)],
                       implanting_config=ImplantingConfig(
                           control=ImplantingSetting(train=ImplantingGroup(baseline=ImplantingUnit(0.15, [], pos_given_no_motif_prob,
                                                                                                   pos_given_motif_prob),
                                                                           modified=ImplantingUnit(0.15, [], pos_given_no_motif_prob,
                                                                                                   pos_given_motif_prob),
                                                                           seq_count=seq_count),
                                                     test=ImplantingGroup(baseline=ImplantingUnit(0.15, [], pos_given_no_motif_prob,
                                                                                                  pos_given_motif_prob),
                                                                          modified=ImplantingUnit(0.15, [], pos_given_no_motif_prob,
                                                                                                  pos_given_motif_prob),
                                                                          seq_count=seq_count),
                                                     name='control'),
                           batch=ImplantingSetting(train=ImplantingGroup(baseline=ImplantingUnit(0.19, [], 72 / 81, 18 / 19), seq_count=seq_count,
                                                                         modified=ImplantingUnit(0.11, batch_effect_genes, 8 / 89, 2 / 11)),
                                                   test=ImplantingGroup(baseline=ImplantingUnit(0.15, [], pos_given_no_motif_prob,
                                                                                                pos_given_motif_prob), seq_count=seq_count,
                                                                        modified=ImplantingUnit(0.15, batch_effect_genes,
                                                                                                pos_given_no_motif_prob, pos_given_motif_prob)),
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
