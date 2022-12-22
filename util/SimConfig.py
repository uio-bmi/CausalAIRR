from dataclasses import dataclass
from typing import List

from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation
from immuneML.simulation.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting
from immuneML.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting
from immuneML.simulation.signal_implanting_strategy.ImplantingComputation import ImplantingComputation


@dataclass
class ImplantingUnit:
    implanting_prob: float
    skip_genes: List[str]


@dataclass
class ImplantingGroup:
    baseline: ImplantingUnit
    modified: ImplantingUnit
    seq_count: int
    name: str


@dataclass
class ImplantingConfig:
    control: ImplantingGroup
    batch: ImplantingGroup


@dataclass
class SimConfig:
    k: int
    repetitions: int
    p_noise: float
    olga_model_name: str
    signal: Signal
    fdrs: List[float]
    implanting_config: ImplantingConfig
    sequence_encoding: str = "continuous_kmer"


def make_signal(motif_seeds: List[str], seq_position_weights: dict = None):
    return Signal("signal", motifs=[Motif(f"m{i}", GappedKmerInstantiation(), seed) for i, seed in enumerate(motif_seeds)],
                  implanting_strategy=HealthySequenceImplanting(implanting=GappedMotifImplanting(),
                                                                implanting_computation=ImplantingComputation.ROUND,
                                                                sequence_position_weights=seq_position_weights))
