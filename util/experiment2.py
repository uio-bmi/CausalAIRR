from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation
from immuneML.simulation.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting
from immuneML.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting
from immuneML.simulation.signal_implanting_strategy.ImplantingComputation import ImplantingComputation


def make_immune_state_signal(signal_name: str = "immune_state") -> Signal:
    motif1 = Motif(identifier="motif1", seed="EQY",
                   instantiation=GappedKmerInstantiation(hamming_distance_probabilities={0: 0.5, 1: 0.5}))

    motif2 = Motif(identifier="motif2", seed="QPR",
                   instantiation=GappedKmerInstantiation(hamming_distance_probabilities={0: 0.8, 1: 0.2}))

    signal1 = Signal(identifier=signal_name, motifs=[motif1, motif2],
                     implanting_strategy=HealthySequenceImplanting(sequence_position_weights={109: 0.5, 110: 0.5},
                                                                   implanting_computation=ImplantingComputation.ROUND,
                                                                   implanting=GappedMotifImplanting()))

    return signal1


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
