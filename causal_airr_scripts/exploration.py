from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.encoding_reports.FeatureComparison import FeatureComparison
from immuneML.reports.encoding_reports.FeatureDistribution import FeatureDistribution
from immuneML.reports.encoding_reports.FeatureValueBarplot import FeatureValueBarplot

from .olga_util import gen_olga_sequences
from .kmer_enrichment import get_kmer_presence_from_sequences
from .dataset_util import setup_path, get_dataset_from_dataframe


def gene_distribution_in_olga(sequences: pd.DataFrame, path: Path):
    sorted_seqs = sequences.sort_values(by=['v_call'])
    fig = px.histogram(sorted_seqs, 'v_call', template='plotly_white', title='V gene distribution', histnorm='probability')
    fig.write_html(path / f'v_gene_distribution_humanTRB.html')


def plot_kmer_frequencies(sequences: pd.DataFrame, k: int, path: Path, name: str):
    kmers_per_sequence = get_kmer_presence_from_sequences(sequences, k=k)
    data = pd.DataFrame(kmers_per_sequence.transpose().sum(axis=1), columns=['count'])
    fig = go.Figure([go.Bar(x=data.index.values, y=data['count'].values / sequences.shape[0])])
    fig.update_layout(title=f'{k}-mer distribution', template='plotly_white')
    fig.write_html(path / f"{k}-mer distribution_{name}.html")


def compare_kmer_frequencies(baseline_seqs: pd.DataFrame, modified_seqs: pd.DataFrame, k: int, path: Path, name: str):
    seqs1_kmers = get_kmer_presence_from_sequences(baseline_seqs, k)
    data1 = pd.DataFrame(seqs1_kmers.transpose().sum(axis=1), columns=['count'])

    seqs2_kmers = get_kmer_presence_from_sequences(modified_seqs, k)
    data2 = pd.DataFrame(seqs2_kmers.transpose().sum(axis=1), columns=['count'])

    missing_kmers = list(set(data1.index.values).difference(set(data2.index.values)))
    data2 = pd.concat([data2, pd.DataFrame(data={'count': [0 for _ in range(len(missing_kmers))]}, index=missing_kmers)])
    data2.sort_index(inplace=True)
    missing_kmers = list(set(data2.index.values).difference(set(data1.index.values)))
    data1 = pd.concat([data1, pd.DataFrame(data={'count': [0 for _ in range(len(missing_kmers))]}, index=missing_kmers)])
    data1.sort_index(inplace=True)

    fig = go.Figure([go.Scatter(x=np.log(data1['count'] / baseline_seqs.shape[0]), y=np.log(data2['count'] / modified_seqs.shape[0]),
                                mode='markers', opacity=0.7, text=data1.index)])
    fig.update_layout(template='plotly_white', xaxis_title="baseline", yaxis_title="modified",
                      title="K-mer frequency across sequences from the baseline vs from the modified model")
    fig.write_html(path / f"{k}-mer distribution_comparison_{name}.html")


def run_distribution_report(sequences: pd.DataFrame, path: Path):
    dataset = get_dataset_from_dataframe(sequences, path / 'seqs.tsv')
    encoder = KmerFrequencyEncoder.build_object(dataset, **{'normalization_type': 'l2', 'reads': 'unique', 'sequence_encoding': 'continuous_kmer',
                                                            'k': 3, 'scale_to_unit_variance': True, 'scale_to_zero_mean': False,
                                                            'sequence_type': 'amino_acid'})

    dataset = encoder.encode(dataset, EncoderParams(result_path=path / 'encoded', label_config=None, learn_model=True, encode_labels=False))
    report = FeatureDistribution(dataset, path / 'report', name='kmer_frequency', )
    report.generate_report()


def combine_two_dataframes(baseline_seqs: pd.DataFrame, modified_seqs: pd.DataFrame) -> pd.DataFrame:
    combined_seqs = baseline_seqs.copy(deep=True)
    combined_seqs['status'] = 'baseline'
    modified_with_status = modified_seqs.copy(deep=True)
    modified_with_status['status'] = 'modified'
    combined_seqs = pd.concat([combined_seqs, modified_with_status])
    return combined_seqs


def run_comparison_report(baseline_seqs: pd.DataFrame, modified_seqs: pd.DataFrame, path: Path, sequence_encoding: str = 'continuous_kmer',
                          keep_fraction=0.5):
    combined_seqs = combine_two_dataframes(baseline_seqs, modified_seqs)
    dataset = get_dataset_from_dataframe(combined_seqs, path / 'seq_dataset.tsv')

    encoder = KmerFrequencyEncoder.build_object(dataset, **{'normalization_type': 'l2', 'reads': 'unique', 'sequence_encoding': sequence_encoding,
                                                            'k': 3, 'scale_to_unit_variance': True, 'scale_to_zero_mean': False,
                                                            'sequence_type': 'amino_acid'})

    dataset = encoder.encode(dataset, EncoderParams(result_path=path / 'encoded', learn_model=True, encode_labels=True,
                                                    label_config=LabelConfiguration([Label('status', ['baseline', 'modified'])])))

    report = FeatureComparison(dataset, path / 'report', name='kmer_frequency_comparison', comparison_label='status', show_error_bar=False,
                               keep_fraction=keep_fraction)
    report.generate_report()


def plot_immuneml_kmer_frequencies(sequences: pd.DataFrame, k: int, positional: bool, path: Path, col_mapping: dict):
    dataset = get_dataset_from_dataframe(sequences, path / 'seq_dataset.tsv', col_mapping, {})

    encoder = KmerFrequencyEncoder.build_object(dataset, **{'normalization_type': 'l2', 'reads': 'unique',
                                                            'sequence_encoding': f'{"imgt_" if positional else ""}continuous_kmer',
                                                            'k': k, 'scale_to_unit_variance': False, 'scale_to_zero_mean': False,
                                                            'sequence_type': 'amino_acid'})

    dataset = encoder.encode(dataset, EncoderParams(result_path=path / 'encoded', learn_model=True, encode_labels=False,
                                                    label_config=LabelConfiguration()))

    dataset = filter_top_n_from_encoded_data(dataset, 100, max_position=108)

    report = FeatureValueBarplot(dataset, path / 'report',
                                 name=f'{"imgt_" if positional else ""}_{k}-mer_frequency', y_title="frequency", x_title=f'{k}-mers',
                                 show_error_bar=False)
    report.generate_report()


def filter_top_n_from_encoded_data(dataset, n, min_position=0, max_position=200):
    avg_freqs = np.array(dataset.encoded_data.examples.mean(axis=0)).flatten()
    max_avg_freq_indices = np.argsort(avg_freqs)[::-1][:n]

    indices_of_start_of_seq_kmers = [i for i in range(len(dataset.encoded_data.feature_names))
                                     if min_position <= float(dataset.encoded_data.feature_names[i].split("-")[1]) <= max_position
                                     and i in max_avg_freq_indices]
    dataset.encoded_data.examples = dataset.encoded_data.examples[:, indices_of_start_of_seq_kmers]
    dataset.encoded_data.feature_names = np.array(dataset.encoded_data.feature_names)[indices_of_start_of_seq_kmers].tolist()
    dataset.encoded_data.feature_annotations = dataset.encoded_data.feature_annotations.iloc[indices_of_start_of_seq_kmers, :]
    return dataset


if __name__ == "__main__":
    # gene_distribution_in_olga(gen_olga_sequences('humanTRB', [], 10000), Path("../exploratory_results/"))

    # sequences = gen_olga_sequences('humanTRB', ['TRBV20'], 5000)
    # plot_immuneml_kmer_frequencies(sequences, 3, True, setup_path(Path("../exploratory_results/positional_kmer_freqs_without_TRBV20")),
    #                                {0: 'sequence_aas', 1: 'v_genes', 2: 'j_genes'})

    run_comparison_report(modified_seqs=gen_olga_sequences('humanTRB', ['TRBV20', "TRBV5-1", "TRBV27", "TRBV24"], 3000),
                          baseline_seqs=gen_olga_sequences('humanTRB', None, 3000),
                          path=setup_path(Path("../exploratory_results/null_vs_TRBV20_5-1_27_24_removed_comparison")),
                          sequence_encoding='continuous_kmer', keep_fraction=0.3)
