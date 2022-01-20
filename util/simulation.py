import uuid
from pathlib import Path

import yaml
from immuneML.simulation.implants.Signal import Signal
from immuneML.util.PathBuilder import PathBuilder
import numpy as np

from util.implanting import make_repertoire_with_signal, make_repertoire_without_signal
from util.experiment2 import make_exp_protocol_signal
from util.dataset_util import make_olga_repertoire


def get_immune_state(p: float) -> bool:
    return bool(np.random.binomial(n=1, p=p))


def get_hospital(p) -> str:
    return "hospital1" if np.random.binomial(n=1, p=p) > 0.5 else "hospital2"


def get_exp_protocol(hospital: str) -> int:
    protocol_id = 1 if hospital == "hospital1" else 2
    return protocol_id


def get_selection(hospital: str, immune_state: bool) -> bool:
    if hospital == "hospital1":
        if immune_state:
            return bool(np.random.binomial(1, 0.8))
        else:
            return bool(np.random.binomial(1, 0.2))
    if hospital == "hospital2":
        if immune_state:
            return bool(np.random.binomial(1, 0.2))
        else:
            return bool(np.random.binomial(1, 0.8))


def get_repertoire(immune_state: bool, experimental_protocol_id: int, path: Path, sequence_count: int,
                   immune_state_signal: Signal, immune_state_implanting_rate: float, protocol_implanting_rate: float) -> str:
    PathBuilder.build(path)

    # make OLGA repertoire from the default OLGA TCRB model
    repertoire = make_olga_repertoire(path=path, sequence_count=sequence_count, seed=str(uuid.uuid4().hex))

    # implant a signal in the repertoire based on the immune state
    if immune_state:
        repertoire = make_repertoire_with_signal(repertoire=repertoire, signal=immune_state_signal,
                                                 result_path=path / "immuneML_with_signal/",
                                                 repertoire_implanting_rate=immune_state_implanting_rate)
    else:
        repertoire = make_repertoire_without_signal(repertoire=repertoire, signal_name=immune_state_signal.id,
                                                    result_path=path / "immuneML_with_signal/")

    # simulate experimental protocol signal in the repertoire

    exp_protocol_signal = make_exp_protocol_signal(protocol_id=experimental_protocol_id, signal_name="experimental_protocol")

    repertoire = make_repertoire_with_signal(repertoire=repertoire, signal=exp_protocol_signal,
                                             result_path=path / "immuneML_with_protocol/",
                                             repertoire_implanting_rate=protocol_implanting_rate)

    update_protocol_metadata(repertoire.metadata_filename, experimental_protocol_id)

    return repertoire.data_filename


def update_protocol_metadata(metadata_filename: Path, experimental_protocol_id: int):

    with metadata_filename.open('r') as file:
        metadata = yaml.safe_load(file)

    metadata['experimental_protocol'] = experimental_protocol_id

    with metadata_filename.open('w') as file:
        yaml.dump(metadata, file)
