from pathlib import Path

import pandas as pd
from immuneML.simulation.implants.Signal import Signal

from causal_airr_scripts.dataset_util import write_to_file
from causal_airr_scripts.experiment3.SimConfig import ImplantingGroup, ImplantingUnit
from causal_airr_scripts.implanting import implant_in_sequences
from causal_airr_scripts.olga_util import gen_olga_sequences


def generate_sequences(olga_model_name: str, impl_group: ImplantingGroup, signal: Signal, p_noise: float, path: Path, setting_name: str) \
                       -> pd.DataFrame:

    seq_path = path / "sequences.tsv"
    gen_sequences_for_implanting_unit(impl_group.baseline, olga_model_name, round(impl_group.seq_count/2), signal, 0, p_noise, seq_path, setting_name)
    gen_sequences_for_implanting_unit(impl_group.modified, olga_model_name, round(impl_group.seq_count/2), signal, 1, p_noise, seq_path, setting_name)

    return pd.read_csv(seq_path, sep="\t")


def gen_sequences_for_implanting_unit(implanting_unit: ImplantingUnit, olga_model_name: str, seq_count: int, signal: Signal, batch_name: int,
                                      p_noise: float, seq_path: Path, setting_name: str):

    sequences = gen_olga_sequences(olga_model_name, implanting_unit.skip_genes, seq_count)
    sequences = implant_in_sequences(sequences, signal, implanting_unit.implanting_prob, p_noise)
    sequences = add_column(sequences, 'batch', batch_name)
    sequences = add_column(sequences, 'setting', setting_name)
    write_to_file(sequences, seq_path)


def add_column(sequences: pd.DataFrame, col_name: str, col_val):
    sequences[col_name] = col_val
    return sequences
