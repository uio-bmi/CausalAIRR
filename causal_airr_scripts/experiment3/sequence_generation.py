from pathlib import Path
from re import Pattern

import pandas as pd
from immuneML.simulation.implants.Signal import Signal

from causal_airr_scripts.dataset_util import write_to_file
from causal_airr_scripts.experiment3.SimConfig import ImplantingGroup, ImplantingUnit
from causal_airr_scripts.implanting import implant_in_sequences
from causal_airr_scripts.olga_util import gen_olga_sequences


def generate_sequences(olga_model_name: str, impl_group: ImplantingGroup, signal: Signal, path: Path, setting_name: str, skip_motifs: Pattern = None) \
                       -> pd.DataFrame:

    seq_path = path / "sequences.tsv"
    gen_sequences_for_implanting_unit(impl_group.baseline, olga_model_name, round(impl_group.seq_count/2), signal, 0, seq_path, setting_name, skip_motifs)
    gen_sequences_for_implanting_unit(impl_group.modified, olga_model_name, round(impl_group.seq_count/2), signal, 1, seq_path, setting_name, skip_motifs)

    return pd.read_csv(seq_path, sep="\t")


def gen_sequences_for_implanting_unit(implanting_unit: ImplantingUnit, olga_model_name: str, seq_count: int, signal: Signal, batch_name: int,
                                      seq_path: Path, setting_name: str, skip_motifs: Pattern):

    sequences = gen_olga_sequences(olga_model_name, implanting_unit.skip_genes, seq_count, skip_motifs=skip_motifs)
    sequences = implant_in_sequences(sequences, signal, implanting_unit)
    sequences = add_column(sequences, 'batch', batch_name)
    sequences = add_column(sequences, 'setting', setting_name)
    write_to_file(sequences, seq_path)


def add_column(sequences: pd.DataFrame, col_name: str, col_val):
    sequences[col_name] = col_val
    return sequences
