import copy
from dataclasses import dataclass, field
from typing import List

from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation
from immuneML.simulation.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting
from immuneML.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting
from immuneML.simulation.signal_implanting_strategy.ImplantingComputation import ImplantingComputation

from causal_airr_scripts.implanting import make_signal


@dataclass
class ImplantingUnit:
    label_implanting_prob: float
    label_given_no_motif_prob: float
    label_given_motif_prob: float
    batch_implanting_prob: float = 0.


@dataclass
class ImplantingGroup:
    baseline: ImplantingUnit
    modified: ImplantingUnit
    seq_count: int

    def to_dict(self) -> dict:
        self_dict = copy.deepcopy(vars(self))
        self_dict['baseline'] = copy.deepcopy(vars(self.baseline)) if self.baseline is not None else {}
        self_dict['modified'] = copy.deepcopy(vars(self.modified)) if self.modified is not None else {}
        return self_dict


@dataclass
class ImplantingSetting:
    train: ImplantingGroup
    test: ImplantingGroup
    name: str

    def to_dict(self) -> dict:
        return {'name': self.name, 'train': self.train.to_dict() if self.train is not None else {},
                'test': self.test.to_dict() if self.test is not None else {}}

    @classmethod
    def from_dict(cls, group: dict):
        return ImplantingSetting(**{**group, **{'baseline': ImplantingUnit(**group['baseline']),
                                                'modified': ImplantingUnit(**group['modified'])}})


@dataclass
class ImplantingConfig:
    control: ImplantingSetting
    batch: ImplantingSetting

    def to_dict(self) -> dict:
        return {'control': self.control.to_dict(), 'batch': self.batch.to_dict()}

    @classmethod
    def from_dict(cls, config: dict):
        return ImplantingConfig(control=ImplantingSetting.from_dict(config['control']),
                                batch=ImplantingSetting.from_dict(config['batch']))


@dataclass
class SimConfig:
    k: int
    repetitions: int
    olga_model_name: str
    signal: dict
    batch_signal: dict
    implanting_config: ImplantingConfig
    batch_corrections: list
    sequence_encoding: str = "continuous_kmer"

    def to_dict(self) -> dict:
        config = copy.deepcopy(vars(self))
        config['implanting_config'] = self.implanting_config.to_dict()

        config['batch_corrections'] = copy.deepcopy(
            [f"{obj.__class__.__name__}_{obj.alpha if hasattr(obj, 'alpha') else ''}" for obj in self.batch_corrections])

        return config

    @classmethod
    def from_dict(cls, config: dict):
        signal = make_signal(list(config['signal']['motifs'].keys()), config['signal']['sequence_positions'],
                             hamming_dist_weights=list(config['signal']['motifs'].values())[0]['hamming_dist'],
                             position_weights=list(config['signal']['motifs'].values())[0]['position_weights'])

        return SimConfig(**{**config, **{'signal': signal, 'implanting_config': ImplantingConfig.from_dict(config['implanting_config'])}})
