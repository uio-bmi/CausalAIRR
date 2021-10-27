import shutil
from pathlib import Path
from typing import List

import pandas as pd
from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation
from immuneML.simulation.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting
from immuneML.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting
from immuneML.simulation.signal_implanting_strategy.ImplantingComputation import ImplantingComputation
from immuneML.util.PathBuilder import PathBuilder


def make_immune_signal(signal_name: str = "immune_state") -> Signal:
    motif1 = Motif(identifier="motif1", seed="ADR",
                   instantiation=GappedKmerInstantiation(hamming_distance_probabilities={0: 0.5, 1: 0.5}))

    motif2 = Motif(identifier="motif2", seed="ATS",
                   instantiation=GappedKmerInstantiation(hamming_distance_probabilities={0: 0.5, 1: 0.5}))

    signal = Signal(identifier=signal_name, motifs=[motif1, motif2],
                    implanting_strategy=HealthySequenceImplanting(sequence_position_weights={109: 0.5, 110: 0.5},
                                                                  implanting_computation=ImplantingComputation.POISSON,
                                                                  implanting=GappedMotifImplanting()))

    return signal


def make_exp_protocol_signal(protocol_id: int = 1, signal_name: str = "experimental_protocol_1"):

    if protocol_id == 1:
        seed = "QHF"
    elif protocol_id == 2:
        seed = "EAF"
    else:
        raise ValueError("Protocol id can only be 1 or 2 for now.")

    motif1 = Motif(identifier="motif1", seed=seed,
                   instantiation=GappedKmerInstantiation(hamming_distance_probabilities={0: 0.5, 1: 0.5}))

    signal = Signal(identifier=signal_name, motifs=[motif1],
                    implanting_strategy=HealthySequenceImplanting(sequence_position_weights={114: 0.5, 115: 0.5},
                                                                  implanting_computation=ImplantingComputation.POISSON,
                                                                  implanting=GappedMotifImplanting()))

    return signal


def make_repertoire_without_signal(repertoire: Repertoire, signal_name: str, path: Path) -> Repertoire:
    new_metadata = {**repertoire.metadata, **{signal_name: False}}
    new_repertoire = Repertoire.build_from_sequence_objects(repertoire.sequences, PathBuilder.build(path),
                                                            metadata=new_metadata)
    new_repertoire.metadata['filename'] = new_repertoire.data_filename.name
    return new_repertoire


def make_repertoire_with_signal(repertoire: Repertoire, signal: Signal, result_path: Path):
    new_repertoire = signal.implant_to_repertoire(repertoire=repertoire, repertoire_implanting_rate=0.5, path=result_path)
    new_repertoire.metadata['filename'] = new_repertoire.data_filename.name
    return new_repertoire


def make_AIRR_dataset(repertoires: List[Repertoire], path: Path, dataset_name: str, signal_names: List[str]):

    assert len(repertoires) > 0, "No repertoires in the list, cannot make dataset."

    PathBuilder.build(path / "tmp")

    metadata_keys = [key for key in repertoires[0].metadata.keys() if key != 'field_list']
    metadata_file = path / f"tmp/{dataset_name}_metadata.csv"

    pd.DataFrame({**{"subject_id": [repertoire.identifier for repertoire in repertoires]},
                  **{key: [repertoire.metadata[key] for repertoire in repertoires] for key in metadata_keys}})\
        .to_csv(path_or_buf=metadata_file, index=False)

    dataset = RepertoireDataset(labels={signal_name: [True, False] for signal_name in signal_names}, repertoires=repertoires,
                                metadata_file=metadata_file, name=dataset_name)

    AIRRExporter.export(dataset, path / "AIRR", RegionType.IMGT_JUNCTION)

    shutil.rmtree(path / "tmp")

    return dataset
