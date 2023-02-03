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


def make_summary(paths: List[Path], result_path: Path):
    dfs = []
    for path in paths:
        df = pd.read_csv(path / 'metrics.tsv', sep='\t')
        df['setting'] = path.parent.name + "_" + path.name
        dfs.append(df)

    write_to_file(pd.concat(dfs, axis=0), result_path / 'summary_metrics.tsv')

    fig = make_subplots(1, dfs[0].shape[1] - 2, subplot_titles=dfs[0].columns.tolist()[:-2], horizontal_spacing=0.05, shared_yaxes=True)

    for index, metric in enumerate(dfs[0].columns):
        if metric not in ['repetition', 'setting']:
            for df_index, df in enumerate(dfs):
                name = paths[df_index].parent.name + "_" + paths[df_index].name
                fig.add_trace(go.Box(name=name, y=df[metric].values.tolist(), boxpoints='all', jitter=0.3, pointpos=-1.8,
                                     opacity=0.5, marker_color=px.colors.sequential.Aggrnyl[df_index]), 1, index + 1)

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

    fig.update_layout(template="plotly_white", legend={'orientation': 'h', 'yanchor': 'bottom', 'xanchor': 'right', 'x': 1, 'y': 1.08})
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
