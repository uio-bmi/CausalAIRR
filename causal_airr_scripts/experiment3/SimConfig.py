import copy
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

    def to_dict(self) -> dict:
        self_dict = copy.deepcopy(vars(self))
        self_dict['baseline'] = vars(self.baseline) if self.baseline is not None else {}
        self_dict['modified'] = vars(self.modified) if self.modified is not None else {}
        return self_dict

    @classmethod
    def from_dict(cls, group: dict):
        return ImplantingGroup(**{**group, **{'baseline': ImplantingUnit(**group['baseline']),
                                              'modified': ImplantingUnit(**group['modified'])}})


@dataclass
class ImplantingConfig:
    control: ImplantingGroup
    batch: ImplantingGroup

    def to_dict(self) -> dict:
        return {
            'control': self.control.to_dict(),
            'batch': self.batch.to_dict()
        }

    @classmethod
    def from_dict(cls, config: dict):
        return ImplantingConfig(control=ImplantingGroup.from_dict(config['control']),
                                batch=ImplantingGroup.from_dict(config['batch']))


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

    def to_dict(self) -> dict:
        config = copy.deepcopy(vars(self))
        config['implanting_config'] = self.implanting_config.to_dict()
        config['signal'] = {'signal_id': self.signal.id,
                            'motifs': {motif.seed: {'hamming_dist': copy.deepcopy(motif.instantiation._hamming_distance_probabilities),
                                                    "position_weights": copy.deepcopy(motif.instantiation.position_weights)}
                                       for motif in self.signal.motifs},
                            'sequence_positions': self.signal.implanting_strategy.sequence_position_weights}
        return config

    @classmethod
    def from_dict(cls, config: dict):
        signal = make_signal(list(config['signal']['motifs'].keys()), config['signal']['sequence_positions'],
                             hamming_dist_weights=list(config['signal']['motifs'].values())[0]['hamming_dist'],
                             position_weights=list(config['signal']['motifs'].values())[0]['position_weights'])

        return SimConfig(**{**config, **{'signal': signal, 'implanting_config': ImplantingConfig.from_dict(config['implanting_config'])}})


def make_signal(motif_seeds: List[str], seq_position_weights: dict = None, hamming_dist_weights: dict = None, position_weights: dict = None):
    return Signal("signal", motifs=[Motif(f"m{i}", GappedKmerInstantiation(hamming_distance_probabilities=hamming_dist_weights,
                                                                           position_weights=position_weights), seed)
                                    for i, seed in enumerate(motif_seeds)],
                  implanting_strategy=HealthySequenceImplanting(implanting=GappedMotifImplanting(),
                                                                implanting_computation=ImplantingComputation.ROUND,
                                                                sequence_position_weights=seq_position_weights))
