from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation
from immuneML.simulation.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting
from immuneML.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting
from immuneML.simulation.signal_implanting_strategy.ImplantingComputation import ImplantingComputation


def make_default_signal(signal_name: str = "immune_state") -> Signal:
    motif1 = Motif(identifier="motif1", seed="ADR",
                   instantiation=GappedKmerInstantiation(hamming_distance_probabilities={0: 0.5, 1: 0.5}))

    motif2 = Motif(identifier="motif1", seed="ATS",
                   instantiation=GappedKmerInstantiation(hamming_distance_probabilities={0: 0.5, 1: 0.5}))

    signal = Signal(identifier=signal_name, motifs=[motif1, motif2],
                    implanting_strategy=HealthySequenceImplanting(sequence_position_weights={109: 0.5, 110: 0.5},
                                                                  implanting_computation=ImplantingComputation.POISSON,
                                                                  implanting=GappedMotifImplanting()))

    return signal
