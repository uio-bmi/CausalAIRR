from pathlib import Path
from typing import List

from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation
from immuneML.simulation.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting
from immuneML.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting
from immuneML.simulation.signal_implanting_strategy.ImplantingComputation import ImplantingComputation
from immuneML.util.PathBuilder import PathBuilder


def make_immune_state_signals(signal_name: str = "immune_state") -> List[Signal]:
    motif1 = Motif(identifier="motif1", seed="EQY",
                   instantiation=GappedKmerInstantiation(hamming_distance_probabilities={0: 0.5, 1: 0.5}))

    signal1 = Signal(identifier=signal_name, motifs=[motif1],
                     implanting_strategy=HealthySequenceImplanting(sequence_position_weights={109: 0.5, 110: 0.5},
                                                                   implanting_computation=ImplantingComputation.ROUND,
                                                                   implanting=GappedMotifImplanting()))

    motif2 = Motif(identifier="motif2", seed="QPR",
                   instantiation=GappedKmerInstantiation(hamming_distance_probabilities={0: 0.8, 1: 0.2}))

    signal2 = Signal(identifier=signal_name, motifs=[motif2],
                     implanting_strategy=HealthySequenceImplanting(sequence_position_weights={109: 0.5, 110: 0.5},
                                                                   implanting_computation=ImplantingComputation.ROUND,
                                                                   implanting=GappedMotifImplanting()))

    return [signal1, signal2]


def make_confounder_signal(signal_name: str = "confounder"):
    motif1 = Motif(identifier="motif1", seed="ADR",
                   instantiation=GappedKmerInstantiation(hamming_distance_probabilities={0: 0.9, 1: 0.1}))

    signal = Signal(identifier=signal_name, motifs=[motif1],
                    implanting_strategy=HealthySequenceImplanting(sequence_position_weights={105: 0.7, 106: 0.3},
                                                                  implanting_computation=ImplantingComputation.ROUND,
                                                                  implanting=GappedMotifImplanting()))

    return signal


def make_repertoire_without_signal(repertoire: Repertoire, signal_name: str, result_path: Path) -> Repertoire:
    new_metadata = {**repertoire.metadata, **{signal_name: False}}
    new_repertoire = Repertoire.build_from_sequence_objects(repertoire.sequences, PathBuilder.build(result_path),
                                                            metadata=new_metadata)
    new_repertoire.metadata['filename'] = new_repertoire.data_filename.name
    return new_repertoire


def make_repertoire_with_signal(repertoire: Repertoire, signal: Signal, result_path: Path, repertoire_implanting_rate: float):
    PathBuilder.build(result_path)
    new_repertoire = signal.implant_to_repertoire(repertoire=repertoire, repertoire_implanting_rate=repertoire_implanting_rate, path=result_path)
    new_repertoire.metadata['filename'] = new_repertoire.data_filename.name
    return new_repertoire
