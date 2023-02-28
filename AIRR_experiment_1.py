from datetime import datetime
from pathlib import Path

from causal_airr_scripts.dataset_util import setup_path
from causal_airr_scripts.experiment1.experiment1 import Experiment1, Exp1Config
from causal_airr_scripts.plotting import plot_balanced_error_rate
from causal_airr_scripts.util import prepare_namespace


def main(namespace):
    result_path = setup_path(Path(namespace.result_path) / str(datetime.now()).replace(" ", "_"))
    immune_signal = dict(motif_seeds=["YEQ", "PQH", "LFF"], seq_position_weights={108: 0.5, 109: 0.5}, hamming_dist_weights={1: 0.8, 0: 0.2},
                         position_weights={1: 1.}, signal_name="immune_state")
    confounder_signal = dict(motif_seeds=['CAA'], hamming_dist_weights={1: 0.8, 0: 0.2}, position_weights={1: 1}, seq_position_weights={104: 1.},
                             signal_name='confounder_signal')

    sequence_count = 500
    train_example_count = 500
    test_example_count = 500
    repetitions = 30
    immune_state_p_conf1, immune_state_p_conf2 = 0.82, 0.18
    immune_state_implanting_rate, confounder_implanting_rate = 0.01, 0.2

    experiments = [
        Experiment1("1a", setup_path(result_path / '1a'), repetitions=repetitions, num_processes=namespace.num_processes,
                    config=Exp1Config(train_example_count=train_example_count, test_example_count=test_example_count,
                                      immune_state_p_conf1=immune_state_p_conf1, immune_state_p_conf2=immune_state_p_conf2,
                                      confounder_p_train=0.8, confounder_p_test=0.8,
                                      immune_state_implanting_rate=immune_state_implanting_rate, confounder_implanting_rate=confounder_implanting_rate,
                                      sequence_count=sequence_count, immune_signal=immune_signal, confounder_signal=confounder_signal)),
        Experiment1("1b", setup_path(result_path / '1b'), repetitions=repetitions, num_processes=namespace.num_processes,
                    config=Exp1Config(train_example_count=train_example_count, test_example_count=test_example_count,
                                      immune_state_p_conf1=immune_state_p_conf1, immune_state_p_conf2=immune_state_p_conf2,
                                      confounder_p_train=0.8, confounder_p_test=0.6,
                                      immune_state_implanting_rate=immune_state_implanting_rate, confounder_implanting_rate=confounder_implanting_rate,
                                      sequence_count=sequence_count, immune_signal=immune_signal, confounder_signal=confounder_signal)),
        Experiment1("1c_option1", setup_path(result_path / '1c_option1'), repetitions=repetitions, num_processes=namespace.num_processes,
                    config=Exp1Config(train_example_count=train_example_count, test_example_count=test_example_count,
                                      immune_state_p_conf1=0.82, immune_state_p_conf2=0.18,
                                      confounder_p_train=0.8, confounder_p_test=0.0,
                                      immune_state_implanting_rate=immune_state_implanting_rate, confounder_implanting_rate=confounder_implanting_rate,
                                      sequence_count=sequence_count, immune_signal=immune_signal, confounder_signal=confounder_signal)),
        Experiment1("1c_option2", setup_path(result_path / '1c_option2'), repetitions=repetitions, num_processes=namespace.num_processes,
                    config=Exp1Config(train_example_count=train_example_count, test_example_count=test_example_count,
                                      immune_state_p_conf1=immune_state_p_conf1, immune_state_p_conf2=immune_state_p_conf2,
                                      confounder_p_train=0.8, confounder_p_test=0.01,
                                      immune_state_implanting_rate=immune_state_implanting_rate, confounder_implanting_rate=confounder_implanting_rate,
                                      sequence_count=sequence_count, immune_signal=immune_signal, confounder_signal=confounder_signal))
    ]

    for experiment in experiments:
        results = experiment.run()
        plot_balanced_error_rate(results, result_path / experiment.name, show_figure=False)


if __name__ == "__main__":
    namespace = prepare_namespace()
    main(namespace)
