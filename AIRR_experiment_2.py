from datetime import datetime
from pathlib import Path

from causal_airr_scripts.dataset_util import setup_path
from causal_airr_scripts.experiment2.experiment2 import Experiment2, Exp2Config
from causal_airr_scripts.plotting import plot_balanced_error_rate
from causal_airr_scripts.util import prepare_namespace


def main(namespace):
    result_path = setup_path(Path(namespace.result_path) / str(datetime.now()).replace(" ", "_"))
    immune_signal = dict(motif_seeds=["YEQ", "PQH", "LFF"], seq_position_weights={108: 0.5, 109: 0.5}, hamming_dist_weights={1: 0.8, 0: 0.2},
                         position_weights={1: 1.}, signal_name="immune_state")

    experiments = [
        Experiment2("2a", setup_path(result_path / '2a'), repetitions=5, num_processes=namespace.num_processes,
                    config=Exp2Config(train_example_count=200, test_example_count=100, p_immune_state=0.5, p_hospital=0.5,
                                      immune_state_implanting_rate=0.01, sequence_count=500, immune_signal=immune_signal,
                                      protocol_implanting_rate=0.04, protocol_signal_name='protocol')),
        Experiment2("2b", setup_path(result_path / '2b'), repetitions=5, num_processes=namespace.num_processes,
                    config=Exp2Config(train_example_count=200, test_example_count=100, p_immune_state=0.5, p_hospital=0.5,
                                      immune_state_implanting_rate=0.0, sequence_count=500, immune_signal=immune_signal,
                                      protocol_implanting_rate=0.04, protocol_signal_name='protocol'))
    ]

    for experiment in experiments:
        results = experiment.run()
        plot_balanced_error_rate(results, result_path / experiment.name, show_figure=False)


if __name__ == "__main__":
    namespace = prepare_namespace()
    main(namespace)
