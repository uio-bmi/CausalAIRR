from pathlib import Path

import numpy as np
from immuneML.simulation.implants.Signal import Signal
from immuneML.util.PathBuilder import PathBuilder

from util.dataset_util import make_olga_repertoire
from util.implanting import make_repertoire_with_signal, make_repertoire_without_signal


def get_immune_state(confounder: bool, p_conf1: float, p_conf2: float) -> bool:
    return bool(np.random.binomial(n=1, p=p_conf2) if confounder else np.random.binomial(n=1, p=p_conf1))


def get_confounder(p) -> str:
    return "C1" if np.random.binomial(n=1, p=p) > 0.5 else "C2"


def get_repertoire(immune_state: bool, confounder: str, path: Path, sequence_count: int, immune_state_signal: Signal, confounding_signal: Signal,
                   seed: int, immune_state_implanting_rate: float, confounding_implanting_rate: float) -> Path:
    PathBuilder.build(path)

    # make olga repertoire using the default model
    naive_repertoire = make_olga_repertoire(path=path, sequence_count=sequence_count, seed=seed)

    # implant a signal in the repertoire based on the confounder value
    if confounder == "C1":
        repertoire = make_repertoire_with_signal(naive_repertoire, signal=confounding_signal, result_path=path / 'immuneML_with_confounding/',
                                                 repertoire_implanting_rate=confounding_implanting_rate)
    else:
        repertoire = make_repertoire_without_signal(repertoire=naive_repertoire, signal_name=confounding_signal.id,
                                                    result_path=path / "immuneML_with_confounding/")

    # implant a signal in the repertoire based on the immune state
    if immune_state:
        repertoire = make_repertoire_with_signal(repertoire=repertoire, signal=immune_state_signal, result_path=path / "immuneML_with_signal/",
                                                 repertoire_implanting_rate=immune_state_implanting_rate)
    else:
        repertoire = make_repertoire_without_signal(repertoire=repertoire, signal_name=immune_state_signal.id,
                                                    result_path=path / "immuneML_with_signal/")

    return repertoire.data_filename