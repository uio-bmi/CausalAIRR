from pathlib import Path

import numpy as np
from immuneML.simulation.implants.Signal import Signal
from immuneML.util.PathBuilder import PathBuilder

from util.dataset_util import make_olga_repertoire
from util.implanting import make_repertoire_with_signal, make_repertoire_without_signal


def get_immune_state(confounder: bool, p_conf0: float, p_conf1: float) -> bool:
    return bool(np.random.binomial(n=1, p=p_conf1) if confounder else np.random.binomial(n=1, p=p_conf0))


def get_confounder(p) -> str:
    return "C1" if np.random.binomial(n=1, p=p) > 0.5 else "C2"


def get_repertoire(immune_state: bool, confounder: str, confounder_name: str, path: Path, sequence_count: int, signal: Signal, seed: int,
                   repertoire_implanting_rate: float) -> Path:
    PathBuilder.build(path)

    # make olga repertoire by using one of two Olga models based on confounder value
    naive_repertoire = make_olga_repertoire(confounder=confounder, confounder_name=confounder_name, path=path, sequence_count=sequence_count,
                                            seed=seed)

    # implant a signal in the repertoire based on the immune state
    if immune_state:
        repertoire = make_repertoire_with_signal(repertoire=naive_repertoire, signal=signal, result_path=path / "immuneML_with_signal/",
                                                 repertoire_implanting_rate=repertoire_implanting_rate)
    else:
        repertoire = make_repertoire_without_signal(repertoire=naive_repertoire, signal_name=signal.id, result_path=path / "immuneML_with_signal/")

    return repertoire.data_filename
