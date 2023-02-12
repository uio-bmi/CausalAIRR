from datetime import datetime

from sklearn.linear_model import LinearRegression

from causal_airr_scripts.dataset_util import setup_path
from causal_airr_scripts.experiment3.SimConfig import SimConfig, ImplantingConfig, ImplantingSetting, ImplantingUnit, ImplantingGroup
from causal_airr_scripts.experiment3.experiment3 import Experiment3
from causal_airr_scripts.util import write_config, prepare_namespace


def main(namespace):
    run_one_config({'control': ImplantingUnit(0.4, 0.25, 0.875, 0), "name": 'full_run',
                    'batch0': ImplantingUnit(0.16, 3 / 84, 7 / 16, 0.25),
                    'batch1': ImplantingUnit(0.64, 3 / 4, 63 / 64, 0.75),
                    'batch_test': ImplantingUnit(0.4, 0.25, 0.875, batch_implanting_prob=0.5)},
                   5000, namespace.result_path, namespace.num_processes)


def run_one_config(proba_set, sequence_count, result_path, num_processes):
    signal = dict(motif_seeds=["YEQ", "PQH", "LFF"], seq_position_weights={108: 0.5, 109: 0.5}, hamming_dist_weights={1: 0.8, 0: 0.2},
                  position_weights={1: 1.})

    batch_signal = dict(motif_seeds=['CAA'], hamming_dist_weights={1: 0.8, 0: 0.2}, position_weights={1: 1}, seq_position_weights={104: 1.},
                        signal_name='batch_signal')

    config = SimConfig(k=3, repetitions=5, olga_model_name='humanTRB', sequence_encoding='continuous_kmer', signal=signal,
                       batch_corrections=[None, LinearRegression()], batch_signal=batch_signal,
                       implanting_config=ImplantingConfig(
                           control=ImplantingSetting(
                               train=ImplantingGroup(baseline=proba_set['control'], modified=proba_set['control'], seq_count=sequence_count),
                               test=ImplantingGroup(baseline=proba_set['control'], modified=proba_set['control'], seq_count=sequence_count),
                               name='control'),
                           batch=ImplantingSetting(
                               train=ImplantingGroup(baseline=proba_set['batch0'], seq_count=sequence_count, modified=proba_set['batch1']),
                               test=ImplantingGroup(baseline=proba_set['batch_test'], seq_count=sequence_count, modified=proba_set['batch_test']),
                               name='batch')))

    path = setup_path(result_path / f"experiment3_results/AIRR_classification_setup_{proba_set['name']}_seqcount_{sequence_count}_{datetime.now()}")
    write_config(config, path)

    experiment = Experiment3(config, num_processes=num_processes)
    experiment.run(path)


if __name__ == "__main__":
    namespace = prepare_namespace()
    main(namespace)
