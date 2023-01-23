from pathlib import Path
from re import Pattern

import pandas as pd
from immuneML.simulation.implants.Signal import Signal

from causal_airr_scripts.dataset_util import write_to_file
from causal_airr_scripts.experiment3.SimConfig import ImplantingGroup, ImplantingUnit
from causal_airr_scripts.implanting import implant_in_sequences
from causal_airr_scripts.olga_util import gen_olga_sequences


def generate_sequences(olga_model_name: str, impl_group: ImplantingGroup, signal: Signal, path: Path, setting_name: str, skip_motifs: Pattern = None,
                       batch_signal: Signal = None) -> pd.DataFrame:

    seq_path = path / "sequences.tsv"

    for batch_index, impl_unit in enumerate([impl_group.baseline, impl_group.modified]):
        gen_sequences_for_implanting_unit(label_implanting=impl_unit, olga_model_name=olga_model_name,
                                          seq_count=round(impl_group.seq_count / 2), signal=signal, batch_name=batch_index, seq_path=seq_path,
                                          setting_name=setting_name, skip_motifs=skip_motifs, batch_signal=batch_signal)

    return pd.read_csv(seq_path, sep="\t")


def gen_sequences_for_implanting_unit(label_implanting: ImplantingUnit, olga_model_name: str, seq_count: int, signal: Signal, batch_name: int,
                                      seq_path: Path, setting_name: str, skip_motifs: Pattern, batch_signal: Signal):
    sequences = gen_olga_sequences(olga_model_name, skip_v_genes=[], keep_v_genes=[], seq_count=seq_count, skip_motifs=skip_motifs)
    sequences = implant_in_sequences(sequences, signal, label_implanting)
    sequences = implant_in_sequences(sequences, batch_signal, ImplantingUnit(label_implanting.batch_implanting_prob, 0, 1))
    sequences = add_column(sequences, 'batch', batch_name)
    sequences = add_column(sequences, 'setting', setting_name)
    write_to_file(sequences, seq_path)


def add_column(sequences: pd.DataFrame, col_name: str, col_val):
    sequences[col_name] = col_val
    return sequences
