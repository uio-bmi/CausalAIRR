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
    label_implanting_prob: float  # P(label) = P(receptor sequence being specific to the antigen in question)
    label_given_no_motif_prob: float  # P(label | immune motif not present)
    label_given_motif_prob: float  # P(label | immune motif present)
    batch_implanting_prob: float = 0.  # P(batch) = P(receptor sequence containing any motifs of the batch signal)


@dataclass
class ImplantingGroup:
    baseline: ImplantingUnit  # batch 0 -> e.g., coming from experimental protocol 0 without batch signals
    modified: ImplantingUnit  # batch 1 -> e.g., coming from experimental protocol 1 with batch signals
    seq_count: int  # how many AIR sequences to simulate for each of the batches

    def to_dict(self) -> dict:
        self_dict = copy.deepcopy(vars(self))
        self_dict['baseline'] = copy.deepcopy(vars(self.baseline)) if self.baseline is not None else {}
        self_dict['modified'] = copy.deepcopy(vars(self.modified)) if self.modified is not None else {}
        return self_dict


@dataclass
class ImplantingSetting:
    train: ImplantingGroup  # simulation params for training data
    test: ImplantingGroup  # simulation params for test data
    name: str  # user-defined name for the setting, used as a part of the output path

    def to_dict(self) -> dict:
        return {'name': self.name, 'train': self.train.to_dict() if self.train is not None else {},
                'test': self.test.to_dict() if self.test is not None else {}}

    @classmethod
    def from_dict(cls, group: dict):
        return ImplantingSetting(**{**group, **{'baseline': ImplantingUnit(**group['baseline']),
                                                'modified': ImplantingUnit(**group['modified'])}})


@dataclass
class ImplantingConfig:
    control: ImplantingSetting  # how to simulate data with immune signal only (no batch effects)
    batch: ImplantingSetting  # how to simulate data which has immune signal and batch effects

    def to_dict(self) -> dict:
        return {'control': self.control.to_dict(), 'batch': self.batch.to_dict()}

    @classmethod
    def from_dict(cls, config: dict):
        return ImplantingConfig(control=ImplantingSetting.from_dict(config['control']),
                                batch=ImplantingSetting.from_dict(config['batch']))


@dataclass
class SimConfig:
    k: int  # k-mer size (e.g., 3)
    repetitions: int  # how many times to replicate the simulation
    olga_model_name: str  # which model to use from OLGA (AIR simulation tool, here: always humanTRB)
    signal: dict  # which motifs will be a part of the immune signal defining receptor specificity
    batch_signal: dict  # which motifs will be in the batch signal
    implanting_config: ImplantingConfig
    batch_corrections: list  # which models to use for batch corrections
    sequence_encoding: str = "continuous_kmer"  # how to represent a sequence (here: AIRRSEQ -> AIR, IRR, RRS, RSE, SEQ)

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
