import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def make_fig_5d(summary_file: str, result_path: str, settings_name_mapping: dict):
    df = pd.read_csv(summary_file, sep='\t')
    overlaps = sorted([col for col in df.columns if 'overlap' in col])
    repetitions = df.groupby('setting').size().loc['batch_linear_reg']

    fig = go.Figure()
    xshift = [-73, 0, 73]

    mapped_overlap_names = [o.replace("_", " ") if o != 'overlap_0' else 'no overlap' for o in overlaps]
    x = list(itertools.chain.from_iterable([[overlap for _ in range(repetitions)] for overlap in mapped_overlap_names]))

    index = 0

    for old_setting_name, new_setting_name in settings_name_mapping.items():
        setting_df = df.loc[[setting == old_setting_name for setting in df['setting']], overlaps]
        y = np.concatenate([setting_df[overlap].values for overlap in overlaps])

        fig.add_trace(go.Box(y=y, x=x, name=new_setting_name, boxpoints='all', pointpos=0, opacity=0.7, jitter=0.2,
                             marker_color=px.colors.qualitative.Dark2[index]))

        for overlap_index, overlap in enumerate(mapped_overlap_names):
            annotation_y = np.median(setting_df[overlaps[overlap_index]].values).astype(int)
            fig.add_annotation(x=overlap, y=annotation_y, text=str(annotation_y), showarrow=False, yshift=15, font_color='black',
                               xshift=xshift[index])

        index += 1

    fig.update_layout(yaxis={'color': 'black', 'tickfont': {'size': 20}}, template='plotly_white', font_size=20, font_color='black', boxmode='group',
                      legend={'x': 1, 'y': 1, 'yanchor': 'top', 'xanchor': 'right'})
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', color='black', tickfont={'size': 20})
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', color='black', tickfont={'size': 20}, title='number of enriched k-mers')

    fig.write_html(result_path)
    return fig


def make_figure_5b(coefficients_file, result_path, mapping_setting_names: dict):
    df = pd.read_csv(coefficients_file, sep='\t')
    fig = go.Figure()
    index = 0
    repetitions = df.groupby('id').size().loc[list(mapping_setting_names.keys())[0]]

    x = list(itertools.chain.from_iterable([[overlap for _ in range(repetitions)] for overlap in ['no overlap', 'overlap']]))

    xshift = [-100, 0, 100]

    for old_setting_name, new_setting_name in mapping_setting_names.items():
        y_overlap = df.loc[df['id'] == old_setting_name, 'n_overlap'].values
        y_no_overlap = df.loc[df['id'] == old_setting_name, 'n_nonzero'].values - y_overlap

        y = np.concatenate([y_no_overlap, y_overlap])

        fig.add_trace(go.Box(y=y, x=x, name=new_setting_name, boxpoints='all', pointpos=0, opacity=0.7, jitter=0.2,
                             marker_color=px.colors.qualitative.Dark2[index]))

        for tmp_x, tmp_y in {'no overlap': y_no_overlap, 'overlap': y_overlap}.items():
            annotation_y = np.median(tmp_y).astype(int)
            fig.add_annotation(x=tmp_x, y=annotation_y, text=str(annotation_y), showarrow=False, yshift=25, font_color='black', xshift=xshift[index])

        index += 1

    fig.update_layout(yaxis={'color': 'black', 'tickfont': {'size': 20}}, template='plotly_white', font_size=20, font_color='black', boxmode='group',
                      legend={'x': 1, 'y': 1, 'yanchor': 'top', 'xanchor': 'left'})
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', color='black', tickfont={'size': 20})
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', color='black', tickfont={'size': 20}, title='number of enriched genes')

    fig.write_html(result_path)
    return fig


def make_figure_5c(summary_metrics_path, result_path, mapping_setting_names: dict):
    df = pd.read_csv(summary_metrics_path, sep='\t')[['setting', 'balanced_accuracy', 'repetition']]

    fig = go.Figure()
    index = 0

    for old_setting_name, new_setting_name in mapping_setting_names.items():
        y = 1. - df[df['setting'] == old_setting_name]['balanced_accuracy'].values

        fig.add_trace(go.Box(y=y, name=new_setting_name, boxpoints='all', pointpos=0, opacity=0.7, jitter=0.2,
                             marker_color=px.colors.qualitative.Dark2[index]))

        annotation_y = np.median(y)
        fig.add_annotation(x=new_setting_name, y=annotation_y, text=round(annotation_y, 2), showarrow=False, yshift=15, font_color='black')

        index += 1

    _update_layout_and_save(fig, result_path, 'balanced error rate')


def make_figure_5a(metrics_file, result_path, mapping_setting_names):
    df = pd.read_csv(metrics_file, sep='\t')
    df = df.loc[df['metric'] == 'logistic_metrics.balanced_accuracy']

    fig = go.Figure()
    index = 0

    for old_setting_name, new_setting_name in mapping_setting_names.items():
        y = 1. - df.loc[df['type'] == old_setting_name, 'values'].values

        fig.add_trace(go.Box(y=y, name=new_setting_name, boxpoints='all', pointpos=0, opacity=0.7, jitter=0.2,
                             marker_color=px.colors.qualitative.Dark2[index]))

        annotation_y = np.median(y)
        fig.add_annotation(x=new_setting_name, y=annotation_y, text=round(annotation_y, 2), showarrow=False, yshift=15, font_color='black')

        index += 1

    _update_layout_and_save(fig, result_path, y_title='balanced error rate')


def _update_layout_and_save(fig, result_path, y_title, showlegend=False):
    fig.update_layout(yaxis={'color': 'black', 'tickfont': {'size': 20}}, template='plotly_white', font_size=20, font_color='black',
                      boxgap=0, showlegend=showlegend)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', color='black', tickfont={'size': 20})
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', color='black', tickfont={'size': 20}, title=y_title)

    fig.write_html(result_path)
    return fig


if __name__ == "__main__":
    mapping_setting_names = {
        "batch_no_correction": "batch_uncorrected",
        "batch_linear_reg": "batch_corrected",
        "control_no_correction": "control"
    }

    mapping_transcriptomic_setting_names = {
        "uncorrected": 'batch_uncorrected',
        'corrected': 'batch_corrected',
        'no_batch_effects': 'control'
    }

    make_figure_5a("results_exp3/transcriptomics_results/performance_metrics_causalAIRR_transcriptomics_experiment_13032023.tsv", "./exp3_fig5a.html",
                   mapping_transcriptomic_setting_names)
    make_figure_5b("results_exp3/transcriptomics_results/nonzero_coef_stats_causalAIRR_transcriptomics_experiment_13032023.tsv", "./exp3_fig5b.html",
                   mapping_transcriptomic_setting_names)
    make_figure_5c("results_exp3/AIRR_classification_setup_full_run_seqcount_5000_2023-02-21 17:46:40.131096/summary_metrics.tsv", "./exp3_fig5c.html",
                   mapping_setting_names)
    make_fig_5d("results_exp3/AIRR_classification_setup_full_run_seqcount_5000_2023-02-21 17:46:40.131096/summary_enriched_kmers.tsv", "./exp3_fig5d.html",
                mapping_setting_names)
