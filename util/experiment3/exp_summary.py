from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px

from util.dataset_util import setup_path


def make_summary(corrected_path: Path, not_corrected_path: Path, result_path: Path):
    corrected_df = pd.read_csv(corrected_path, sep='\t')
    not_corrected_df = pd.read_csv(not_corrected_path, sep='\t')

    fig = make_subplots(1, corrected_df.shape[1] - 1, subplot_titles=corrected_df.columns.tolist()[:-1], horizontal_spacing=0.05)

    for index, metric in enumerate(corrected_df.columns):
        if metric != 'repetition':
            fig.add_trace(go.Box(name=metric, x=['baseline' for _ in range(corrected_df.shape[0])] + ['corrected' for _ in range(corrected_df.shape[0])],
                                 y=not_corrected_df[metric].values.tolist() + corrected_df[metric].values.tolist(), boxpoints='all', jitter=0.3,
                                 marker_color=px.colors.sequential.Aggrnyl[index]), 1, index+1)

    fig.update_layout(boxmode='group', template='plotly_white', showlegend=False)

    fig.write_html(result_path / 'summary.html')


if __name__ == "__main__":
    make_summary(Path("/Users/milenpa/PycharmProjects/CausalAIRR/experiment3/AIRR_classification_2022-12-21 23:21:32.006013/metrics.tsv"),
                 Path("/Users/milenpa/PycharmProjects/CausalAIRR/experiment3/AIRR_classification_without_correction_2022-12-21 23:30:53.635734/metrics.tsv"),
                 setup_path(Path("/Users/milenpa/PycharmProjects/CausalAIRR/experiment3/exp3_summary_AIRR_classification")))
