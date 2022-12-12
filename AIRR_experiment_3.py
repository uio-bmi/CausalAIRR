import itertools
import logging
from datetime import datetime
from pathlib import Path
import random

import pandas as pd
from immuneML.simulation.implants.Signal import Signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from util.SimConfig import SimConfig, make_signal, ImplantingConfig, ImplantingGroup, ImplantingUnit
from util.dataset_util import setup_path, write_to_file
from util.implanting import implant_in_sequences
from util.kmer_enrichment import find_enriched_kmers, make_contingency_table, compute_p_value, compute_fdr
from util.olga_util import gen_olga_sequences
from util.util import overlaps, get_overlap_length, write_config


def assess_performance(motifs, enriched_kmers, k: int):
    if enriched_kmers.shape[0] > 0:
        kmers_in_motifs = sum([int(kmer in [motif.seed for motif in motifs]) for kmer in enriched_kmers.index]) / enriched_kmers.shape[0]
        kmers_in_partial_motifs = sum([int(any(overlaps(kmer, motif.seed) for motif in motifs)) for kmer in enriched_kmers.index]) / enriched_kmers.shape[
            0]
        recovered_motifs = sum([int(motif.seed in enriched_kmers.index) for motif in motifs]) / len(motifs)
        recovered_partial_motifs = sum([int(any(overlaps(motif.seed, kmer) for kmer in enriched_kmers.index)) for motif in motifs]) / len(motifs)
    else:
        kmers_in_motifs, kmers_in_partial_motifs, recovered_motifs, recovered_partial_motifs = 0, 0, 0, 0

    kmer_counts_per_overlap = {f"overlap_{i}": 0 for i in range(k + 1)}
    for kmer in enriched_kmers.index:
        overlap_length = max([get_overlap_length(kmer, motif.seed) for motif in motifs])
        kmer_counts_per_overlap[f"overlap_{overlap_length}"] += 1

    return {**{"kmers_in_motifs": kmers_in_motifs, "kmers_in_partial_motifs": kmers_in_partial_motifs,
               "recovered_partial_motifs": recovered_partial_motifs, "recovered_motifs": recovered_motifs}, **kmer_counts_per_overlap}


def experiment3_analysis(path, implanting_group: ImplantingGroup, repetition_index: int, sim_config: SimConfig):

    exp_path = setup_path(path / f"{implanting_group.name}_{repetition_index}")

    sequences = generate_sequences(sim_config.olga_model_name, implanting_group, sim_config.signal, sim_config.p_noise, exp_path)

    contingency_table = make_contingency_table(sequences, sim_config.k)
    contingency_with_p_value = compute_p_value(contingency_table)
    write_to_file(contingency_with_p_value, exp_path / f'all_{sim_config.k}mers_with_p_value.tsv')

    results = []
    motifs = sim_config.signal.motifs

    for p_value_threshold in sim_config.p_value_thresholds:
        enriched_kmers = find_enriched_kmers(contingency_with_p_value, p_value_threshold)
        FDR = ""

        write_to_file(enriched_kmers, exp_path / f"enriched_{sim_config.k}mers_{p_value_threshold}.tsv")

        metrics = assess_performance(motifs, enriched_kmers, sim_config.k)
        results.append({**{"k": sim_config.k, "p_value_threshold": p_value_threshold, "group": implanting_group.name, "FDR": FDR,
                           "repetition": repetition_index}, **metrics})
        logging.info(f"Finished for p-value threshold={p_value_threshold}, group={implanting_group.name}, repetition={repetition_index}.")

    return results


def make_noisy_labels(sequences: pd.DataFrame, label: str, p_noise: float) -> pd.DataFrame:
    sequences[label] = [val if random.uniform(0, 1) > p_noise else not val for val in sequences[label]]
    return sequences


def generate_sequences(olga_model_name: str, impl_group: ImplantingGroup, signal: Signal, p_noise: float, path: Path) \
                       -> pd.DataFrame:

    seq_path = path / "sequences.tsv"
    gen_sequences_for_implanting_unit(impl_group.baseline, olga_model_name, impl_group.seq_count, signal, 'baseline_batch', p_noise, seq_path, impl_group.name)
    gen_sequences_for_implanting_unit(impl_group.modified, olga_model_name, impl_group.seq_count, signal, 'modified_batch', p_noise, seq_path, impl_group.name)

    return pd.read_csv(seq_path, sep="\t")


def gen_sequences_for_implanting_unit(implanting_unit: ImplantingUnit, olga_model_name: str, seq_count: int, signal: Signal, batch_name: str,
                                      p_noise: float, seq_path: Path, group_name: str):

    sequences = gen_olga_sequences(olga_model_name, implanting_unit.skip_genes, seq_count)
    sequences = implant_in_sequences(sequences, signal, implanting_unit.implanting_prob)
    sequences = add_column(sequences, 'batch', batch_name)
    sequences = make_noisy_labels(sequences, 'signal', p_noise)
    sequences = add_column(sequences, 'group', group_name)
    write_to_file(sequences, seq_path)


def add_column(sequences: pd.DataFrame, col_name: str, col_val):
    sequences[col_name] = col_val
    return sequences


def plot_proportion_discovered(df: pd.DataFrame, path: Path, k: int):

    fig = make_subplots(1, df['p_value_threshold'].unique().shape[0],
                        subplot_titles=[f"p-value: {pvt:.2e}" for pvt in df['p_value_threshold'].unique()],
                        x_title="overlap length", y_title="number of enriched k-mers", horizontal_spacing=0.05)

    for index, p_value_threshold in enumerate(df['p_value_threshold'].unique()):
        tmp_df = df[df.p_value_threshold == p_value_threshold]

        for gene_index, group in enumerate(tmp_df['group'].unique()):
            tmp_y = tmp_df[tmp_df['group'] == group][[f'overlap_{overlap_length}' for overlap_length in range(k + 1)]]
            y = tmp_y.values.flatten('F')
            x = list(itertools.chain.from_iterable([[i for _ in range(tmp_y.shape[0])] for i in range(k + 1)]))

            fig.add_trace(go.Box(name=str(group), x=x, y=y, opacity=0.7, offsetgroup=group, marker={'opacity': 0.5},
                                 legendgroup=group, showlegend=index == 0, marker_color='#636EFA' if gene_index else '#EF553B'),
                          1, index + 1)

    fig.update_layout(boxmode='group', template="plotly_white", boxgap=0.2, boxgroupgap=0.1,
                      legend={'orientation': 'h', 'yanchor': 'bottom', 'xanchor': 'right', 'x': 1, 'y': 1.08},
                      title=f"Enriched k-mer overlap with true motifs")
    fig.write_html(path / f"summary_boxplot.html")


def do_experiment3_control(sim_config: SimConfig, path: Path, repetition: int):
    return experiment3_analysis(path, sim_config.implanting_config.control, repetition, sim_config)


def do_experiment3_batches(sim_config: SimConfig, path: Path, repetition: int):
    return experiment3_analysis(path, sim_config.implanting_config.batch, repetition, sim_config)


def main(config: SimConfig):
    path = setup_path(f"experiment3/AIRR_{datetime.now()}")
    write_config(config, path)

    results = []

    for repetition in range(1, config.repetitions + 1):
        results += do_experiment3_control(config, setup_path(path / f"control_replication_{repetition}"), repetition)
        results += do_experiment3_batches(config, setup_path(path / f"batches_replication_{repetition}"), repetition)

    results = pd.DataFrame(results)
    write_to_file(results, path / "summary.tsv")
    plot_proportion_discovered(results, path, config.k)


if __name__ == "__main__":

    config = SimConfig(k=3, repetitions=10, p_noise=0.2, olga_model_name='humanTRB',
                       signal=make_signal(motif_seeds=["YEQ", "PQH", "LFF"], seq_position_weights={108: 0.5, 109: 0.5}),
                       p_value_thresholds=sorted([0.01, 0.005, 0.001, 0.0005]),
                       implanting_config=ImplantingConfig(
                           control=ImplantingGroup(baseline=ImplantingUnit(0.5, []),
                                                   modified=ImplantingUnit(0.5, []), name='control', seq_count=10000),
                           batch=ImplantingGroup(baseline=ImplantingUnit(0.9, []), name='batch', seq_count=10000,
                                                 modified=ImplantingUnit(0.1, ["TRBV20", "TRBV5-1", "TRBV24", "TRBV27"]))))

    logging.basicConfig(level=logging.INFO)

    main(config)
