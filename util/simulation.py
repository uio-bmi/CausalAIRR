from pathlib import Path

from immuneML.simulation.implants.Signal import Signal
from immuneML.util.PathBuilder import PathBuilder
import numpy as np

from util.implanting import make_repertoire_with_signal, make_repertoire_without_signal, make_exp_protocol_signal
from util.repertoire_util import make_olga_repertoire


def get_immune_state(p: float) -> bool:
    return bool(np.random.binomial(n=1, p=p))


def get_hospital(p) -> str:
    return "hospital1" if np.random.binomial(n=1, p=p) > 0.5 else "hospital2"


def get_exp_protocol(hospital: str) -> Signal:
    protocol_id = 1 if hospital == "hospital1" else 2
    return make_exp_protocol_signal(protocol_id=protocol_id, signal_name="experimental_protocol")


def get_repertoire(immune_state: bool, experimental_protocol: Signal, path: Path, sequence_count: int,
                   immune_state_signal: Signal, seed: int, repertoire_implanting_rate: float) -> str:
    PathBuilder.build(path)

    # make OLGA repertoire from the default OLGA TCRB model
    repertoire = make_olga_repertoire(path=path, sequence_count=sequence_count, seed=seed, confounder=True, confounder_name="olga_model")

    # implant a signal in the repertoire based on the immune state
    if immune_state:
        repertoire = make_repertoire_with_signal(repertoire=repertoire, signal=immune_state_signal, result_path=path / "immuneML_with_signal/",
                                                 repertoire_implanting_rate=repertoire_implanting_rate)
    else:
        repertoire = make_repertoire_without_signal(repertoire=repertoire, signal_name=immune_state_signal.id, result_path=path / "immuneML_with_signal/")

    # simulate experimental protocol signal in the repertoire
    repertoire = make_repertoire_with_signal(repertoire=repertoire, signal=experimental_protocol, result_path=path / "immuneML_with_protocol/",
                                             repertoire_implanting_rate=repertoire_implanting_rate)

    return repertoire.data_filename
