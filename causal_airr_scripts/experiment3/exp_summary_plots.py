from pathlib import Path
from typing import List

import numpy as np
import plotly.graph_objects as go
import pandas as pd
from immuneML.simulation.implants.Motif import Motif
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
                                 opacity=0.5, marker_color=px.colors.sequential.Aggrnyl[0]), 1, index+1)
            fig.add_trace(go.Box(name='batch_corrected', y=corrected_df[metric].values.tolist(), boxpoints='all', jitter=0.3, pointpos=-1.8,
                                 opacity=0.5, marker_color=px.colors.sequential.Aggrnyl[1]), 1, index+1)
            fig.add_trace(go.Box(name='control', y=control_df[metric].values.tolist(), boxpoints='all', jitter=0.3, pointpos=-1.8, opacity=0.7,
                                 marker_color=px.colors.sequential.Aggrnyl[2]), 1, index + 1)

    fig.update_layout(template='plotly_white', showlegend=False)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    fig.write_html(result_path / 'summary_metrics.html')


def plot_enriched_kmers(path: Path, df: pd.DataFrame, k: int):

    fig = make_subplots(1, k+1, subplot_titles=[f"Motif overlap {overlap}" for overlap in range(k+1)],
                        y_title="number of enriched k-mers", horizontal_spacing=0.05, shared_yaxes=True)

    write_to_file(df, path / f'summary_enriched_kmers.tsv')

    for overlap_length in range(k + 1):
        for setting_index, setting in enumerate(sorted(df['setting'].unique())):
            y = df[df['setting'] == setting][f"overlap_{overlap_length}"]

            fig.add_trace(go.Box(name=str(setting), y=y, opacity=0.7, marker={'opacity': 0.5}, boxpoints='all', jitter=0.3, pointpos=-1.8,
                                 legendgroup=setting, showlegend=overlap_length == 0, marker_color=px.colors.sequential.Aggrnyl[setting_index]),
                          col=overlap_length + 1, row=1)

    fig.update_layout(template="plotly_white", legend={'orientation': 'h', 'yanchor': 'bottom', 'xanchor': 'right', 'x': 1, 'y': 1.08},
                      title=f"Enriched k-mer overlap with true motifs")
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    fig.write_html(path / f"summary_enriched_kmers.html")


def plot_log_reg_coefficients(log_reg: LogisticRegression, feature_names: list, top_n: int, motifs: List[Motif], path: Path):
    df = pd.DataFrame({'coefficient': log_reg.coef_.flatten(), 'feature_name': feature_names})
    write_to_file(df, path / 'log_reg_coefficients.tsv')

    sorted_indices = np.argsort(np.abs(log_reg.coef_.flatten()))[-top_n:]
    df = df.iloc[sorted_indices, :]
    df['motif_overlap'] = [max([get_overlap_length(feature, motif.seed,
                                                   any(key != 0 and val > 0 for key, val in motif.instantiation._hamming_distance_probabilities.items()))
                                for motif in motifs]) for feature in df['feature_name']]

    fig = px.bar(df, x='feature_name', y='coefficient', color='motif_overlap', color_continuous_scale=px.colors.sequential.Aggrnyl)
    fig.update_layout(template='plotly_white', title=f'Logistic regression top {top_n} coefficients')
    fig.write_html(path / f'top_{top_n}_coefficients.html')


def merge_dfs(files, index_name, setting_name) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(file, sep='\t') for file in files], axis=0)
    df.reset_index(inplace=True)
    df.columns.values[0] = index_name
    df['setting'] = setting_name
    return df
