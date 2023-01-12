import re
from pathlib import Path
from re import Pattern
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px

from olga import load_model
from olga.sequence_generation import SequenceGenerationVDJ

DEFAULT_MODEL_FOLDER_MAP = {
    "humanTRA": "human_T_alpha", "humanTRB": "human_T_beta", "humanIGH": "human_B_heavy", "humanIGK": "human_B_kappa",
    "humanIGL": "human_B_lambda", "mouseTRB": "mouse_T_beta", "mouseTRA": "mouse_T_alpha"
}


def gen_olga_sequences(olga_model_name: str, skip_v_genes: list = None, seq_count: int = 0, keep_v_genes: list = None, skip_motifs: Pattern = None,
                       min_length: int = 0):
    olga_model = load_olga_model(olga_model_name)

    sequences = pd.DataFrame(index=np.arange(seq_count), columns=["sequence_aa", "v_call", "j_call"])
    i = 0

    while i < seq_count:
        seq_row = olga_model["gen_model"].gen_rnd_prod_CDR3()

        if keep_sequence(seq_row, skip_v_genes, keep_v_genes, olga_model, min_length, skip_motifs):
            sequences.loc[i] = (seq_row[1], olga_model["v_gene_mapping"][seq_row[2]], olga_model["j_gene_mapping"][seq_row[3]])
            i += 1

    return sequences


def keep_sequence(seq_row, skip_v_genes: list, keep_v_genes: list, olga_model: dict, min_length: int = 0, skip_motifs: Pattern = None):
    return len(seq_row[1]) >= min_length and \
           ((skip_v_genes is not None and not in_list(olga_model['v_gene_mapping'][seq_row[2]], skip_v_genes))
            or (keep_v_genes is not None and in_list(olga_model['v_gene_mapping'][seq_row[2]], keep_v_genes))
            or ((skip_v_genes is None or len(skip_v_genes) == 0) and (keep_v_genes is None or len(keep_v_genes) == 0))) and \
           not re.search(pattern=skip_motifs, string=seq_row[1])


def in_list(gene: str, gene_list: List[str]):
    return any(gene == skip_gene or bool(re.match(f"{skip_gene}[-*]", gene)) for skip_gene in gene_list)


def load_olga_model(olga_model_name, skip_v_genes: list = None) -> dict:
    olga_model_path = Path(load_model.__file__).parent / f"default_models/{DEFAULT_MODEL_FOLDER_MAP[olga_model_name]}"

    gen_model = load_model.GenerativeModelVDJ()
    gen_model.load_and_process_igor_model(str(olga_model_path / "model_marginals.txt"))
    genomic_data = load_model.GenomicDataVDJ()
    genomic_data.load_igor_genomic_data(params_file_name=str(olga_model_path / "model_params.txt"),
                                        V_anchor_pos_file=str(olga_model_path / "V_gene_CDR3_anchors.csv"),
                                        J_anchor_pos_file=str(olga_model_path / "J_gene_CDR3_anchors.csv"))

    v_gene_mapping = [V[0] for V in genomic_data.genV]
    j_gene_mapping = [J[0] for J in genomic_data.genJ]

    sequence_gen_model = SequenceGenerationVDJ(gen_model, genomic_data)

    olga_model = {"gen_model": sequence_gen_model, "v_gene_mapping": v_gene_mapping, "j_gene_mapping": j_gene_mapping}
    if isinstance(skip_v_genes, list) and len(skip_v_genes) > 0:
        olga_model["skip_gene_list"] = skip_v_genes

    return olga_model


def gene_distribution_in_olga(sequences: pd.DataFrame, path: Path):
    sorted_seqs = sequences.sort_values(by=['v_call'])
    fig = px.histogram(sorted_seqs, 'v_call', template='plotly_white', title='V gene distribution', histnorm='probability')
    fig.write_html(path)
