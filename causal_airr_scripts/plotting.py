import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from immuneML.ml_metrics.Metric import Metric

from causal_airr_scripts.util import save_to_yaml


def plot_balanced_error_rate(iml_results: list, result_path, show_figure: bool=True):
    train_states = [res[0] for res in iml_results]
    validation = []
    test = []

    for train_state in train_states:
        assert Metric['BALANCED_ACCURACY'] in train_state.metrics or Metric['BALANCED_ACCURACY'] == train_state.optimization_metric

        selection_state = train_state.assessment_states[0].label_states[train_state.label_configuration.get_labels_by_name()[0]].selection_state

        for hp_setting, hp_item in selection_state.hp_items.items():
            if hp_setting == selection_state.optimal_hp_setting.get_key():
                for item in hp_item:
                    validation.append(1 - item.performance['balanced_accuracy'])

        test.append(1 - train_state.optimal_hp_items['immune_state'].performance['balanced_accuracy'])

    performances = {"validation": validation, "test": test}
    save_to_yaml(performances, result_path / 'balanced_error_rate_performances.yaml')

    figure = plot_error_rate_box(performances, result_path / "validation_vs_test_performance_balanced_error_rate.html")
    if show_figure:
        figure.show()


def plot_error_rate_box(data: dict, result_path):

    figure = go.Figure()
    decimal_count = 3

    i = 0
    for key in data:
        figure.add_box(boxpoints='all', name=key, y=data[key], marker_color=px.colors.diverging.Tealrose[i],
                       text=np.median(data[key]).round(decimal_count), pointpos=0)
        figure.add_annotation(x=key, y=np.median(data[key]), text=str(np.median(data[key]).astype(float).round(decimal_count)), showarrow=False,
                              yshift=15, font_color='black')
        i += 1

    figure.update_layout(yaxis={"title": "balanced error rate", 'color': 'black'}, template='plotly_white', font_size=15, font_color='black')
    figure.update_xaxes(showline=True, linewidth=1, linecolor='black')
    figure.update_yaxes(showline=True, linewidth=1, linecolor='black', color='black')

    figure.write_html(result_path)
    return figure
