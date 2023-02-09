import logging
from pathlib import Path
import random
from typing import List

import numpy as np
import pandas as pd
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
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


def make_immune_state_signal_exp2(signal_name: str = "immune_state") -> Signal:
    motif1 = Motif(identifier="motif1", seed="EQY",
                   instantiation=GappedKmerInstantiation(hamming_distance_probabilities={0: 0.5, 1: 0.5}))

    motif2 = Motif(identifier="motif2", seed="QPR",
                   instantiation=GappedKmerInstantiation(hamming_distance_probabilities={0: 0.8, 1: 0.2}))

    signal = Signal(identifier=signal_name, motifs=[motif1, motif2],
                    implanting_strategy=HealthySequenceImplanting(sequence_position_weights={109: 0.5, 110: 0.5},
                                                                  implanting_computation=ImplantingComputation.ROUND,
                                                                  implanting=GappedMotifImplanting()))

    return signal


def make_exp_protocol_signal(protocol_id: int = 1, signal_name: str = "experimental_protocol"):
    if protocol_id == 1:
        seed = "QHF"
    elif protocol_id == 2:
        seed = "EAF"
    else:
        raise ValueError("Protocol id can only be 1 or 2 for now.")

    motif1 = Motif(identifier="motif1", seed=seed,
                   instantiation=GappedKmerInstantiation(hamming_distance_probabilities={0: 1.}))

    signal = Signal(identifier=signal_name, motifs=[motif1],
                    implanting_strategy=HealthySequenceImplanting(sequence_position_weights={114: 0.5, 115: 0.5},
                                                                  implanting_computation=ImplantingComputation.POISSON,
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


def make_sequence_with_signal(sequence, signal: Signal) -> str:
    if isinstance(sequence, str):
        seq = ReceptorSequence(amino_acid_sequence=sequence, metadata=SequenceMetadata(region_type=RegionType.IMGT_JUNCTION.name))
    else:
        seq = sequence

    motif = random.choice(signal.motifs)

    implanted_seq = signal.implanting_strategy.implant_in_sequence(seq, signal, motif=motif)

    return implanted_seq.amino_acid_sequence


def implant_in_sequences(sequences: pd.DataFrame, signal: Signal, implanting_unit):
    if signal:
        logging.info(f"Implanting for signal {signal.id} with probability {implanting_unit.label_implanting_prob}")
        sequences[signal.id] = False
        sequences[f'{signal.id}_implanted'] = False
        seq_with_motif_count = round(implanting_unit.label_implanting_prob * sequences.shape[0])
        logging.info(f"Sequence count where the motif will be implanted: {seq_with_motif_count}")
        motif_indices = np.random.choice(np.arange(sequences.shape[0]), size=seq_with_motif_count, replace=False)

        for index in motif_indices:
            sequences.at[index, 'sequence_aa'] = make_sequence_with_signal(sequences.at[index, 'sequence_aa'], signal)
            sequences.at[index, f'{signal.id}_implanted'] = True

        seq_with_label_count = round(seq_with_motif_count * implanting_unit.label_given_motif_prob)
        label_indices = np.random.choice(motif_indices, size=seq_with_label_count, replace=False)

        sequences.loc[label_indices, signal.id] = True

        seq_with_label_count = round((sequences.shape[0] - seq_with_motif_count) * implanting_unit.label_given_no_motif_prob)
        label_indices = np.random.choice([i for i in np.arange(sequences.shape[0]) if i not in motif_indices], size=seq_with_label_count,
                                         replace=False)
        sequences.loc[label_indices, signal.id] = True

        sequences = sequences.sample(frac=1)

        logging.info(sequences.head(10))

    return sequences


def make_signal(motif_seeds: List[str], seq_position_weights: dict = None, hamming_dist_weights: dict = None, position_weights: dict = None,
                signal_name: str = "signal"):
    return Signal(signal_name, motifs=[Motif(f"m{i}", GappedKmerInstantiation(hamming_distance_probabilities=hamming_dist_weights,
                                                                              position_weights=position_weights), seed)
                                       for i, seed in enumerate(motif_seeds)],
                  implanting_strategy=HealthySequenceImplanting(implanting=GappedMotifImplanting(),
                                                                implanting_computation=ImplantingComputation.ROUND,
                                                                sequence_position_weights=seq_position_weights))

