import itertools
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.linear_model import LogisticRegression

from causal_airr_scripts.dataset_util import write_to_file
from causal_airr_scripts.util import get_overlap_length


def make_summary(corrected_path: Path, not_corrected_path: Path, control_path: Path, result_path: Path):
    corrected_df = pd.read_csv(corrected_path, sep='\t')
    not_corrected_df = pd.read_csv(not_corrected_path, sep='\t')
    control_df = pd.read_csv(control_path, sep='\t')

    fig = make_subplots(1, corrected_df.shape[1] - 1, subplot_titles=corrected_df.columns.tolist()[:-1], horizontal_spacing=0.05, shared_yaxes=True)

    for index, metric in enumerate(corrected_df.columns):
        if metric != 'repetition':
            fig.add_trace(go.Box(name='batch_baseline', y=not_corrected_df[metric].values.tolist(), boxpoints='all', jitter=0.3, pointpos=-1.8,
                                 marker_color=px.colors.sequential.Aggrnyl[0]), 1, index+1)
            fig.add_trace(go.Box(name='batch_corrected', y=corrected_df[metric].values.tolist(), boxpoints='all', jitter=0.3, pointpos=-1.8,
                                 marker_color=px.colors.sequential.Aggrnyl[1]), 1, index+1)
            fig.add_trace(go.Box(name='control', y=control_df[metric].values.tolist(), boxpoints='all', jitter=0.3, pointpos=-1.8,
                                 marker_color=px.colors.sequential.Aggrnyl[2]), 1, index + 1)

    fig.update_layout(template='plotly_white', showlegend=False)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    fig.write_html(result_path / 'summary_metrics.html')


def plot_enriched_kmers(path: Path, df: pd.DataFrame, k: int):
    for fdr in df['FDR'].unique():

        fig = make_subplots(1, k+1, subplot_titles=[f"Motif overlap {overlap}" for overlap in range(k+1)],
                            y_title="number of enriched k-mers", horizontal_spacing=0.05, shared_yaxes=True)

        tmp_df = df[df['FDR'] == fdr]

        for overlap_length in range(k + 1):
            for group_index, group in enumerate(tmp_df['group'].unique()):
                y = tmp_df[tmp_df['group'] == group][f"overlap_{overlap_length}"]

                fig.add_trace(go.Box(name=str(group), y=y, opacity=0.7, marker={'opacity': 0.5}, boxpoints='all', jitter=0.3, pointpos=-1.8,
                                     legendgroup=group, showlegend=overlap_length == 0, marker_color=px.colors.sequential.Aggrnyl[group_index]),
                              col=overlap_length + 1, row=1)

        fig.update_layout(template="plotly_white", legend={'orientation': 'h', 'yanchor': 'bottom', 'xanchor': 'right', 'x': 1, 'y': 1.08},
                          title=f"Enriched k-mer overlap with true motifs (FDR={fdr})")
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

        fig.write_html(path / f"summary_enriched_kmers_{fdr}.html")


def plot_log_reg_coefficients(log_reg: LogisticRegression, feature_names: list, top_n: int, motifs: list, path: Path):
    df = pd.DataFrame({'coefficient': log_reg.coef_.flatten(), 'feature_name': feature_names})
    write_to_file(df, path / 'log_reg_coefficients.tsv')

    sorted_indices = np.argsort(np.abs(log_reg.coef_.flatten()))[-top_n:]
    df = df.iloc[sorted_indices, :]
    df['motif_overlap'] = [max([get_overlap_length(feature, motif.seed) for motif in motifs]) for feature in df['feature_name']]

    fig = go.Figure()
    for overlap in sorted(df['motif_overlap'].unique()):
        selected_df = df[df['motif_overlap'] == overlap]
        fig.add_trace(go.Bar(x=selected_df['feature_name'], y=selected_df['coefficient'], marker_color=px.colors.sequential.Aggrnyl[overlap],
                             name=f'overlap_{overlap}'))
    fig.update_layout(template='plotly_white', title=f'Logistic regression top {top_n} coefficients')
    fig.write_html(path / f'top_{top_n}_coefficients.html')


def merge_dfs(files, index_name, group_name) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(file, sep='\t') for file in files], axis=0)
    df.reset_index(inplace=True)
    df.columns.values[0] = index_name
    df['group'] = group_name
    return df
