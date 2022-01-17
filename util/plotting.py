from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from immuneML.ml_metrics.Metric import Metric


def plot_balanced_error_rate(iml_result: list, result_path: Path):
    train_state = iml_result[0]
    assert Metric['balanced_accuracy'] in train_state.metrics or Metric['balanced_accuracy'] == train_state.optimization_metric

    selection_state = train_state.assessment_states[0].label_states[train_state.label_configuration.get_labels_by_name()[0]].selection_state

    validation = []
    for hp_setting, hp_item in selection_state.hp_items.items():
        if hp_setting == selection_state.optimal_hp_setting.get_key():
            for item in hp_item:
                validation.append(1 - item.performance['balanced_accuracy'])

    performances = {
        "validation": validation,
        "test": [1 - item.performance['balanced_accuracy'] for item in train_state.optimal_hp_items.items()]
    }

    figure = make_performance_barplot(performances, result_path / "validation_vs_test_performance_balanced_error_rate.html")
    figure.show()


def plot_validation_vs_test_performance(iml_result: list, result_path: Path):
    train_state = iml_result[0]
    metrics = sorted(list(train_state.metrics) + [train_state.optimization_metric], key=lambda m: m.name)

    performances_per_metric = {}

    for metric in metrics:

        selection_state = train_state.assessment_states[0].label_states[train_state.label_configuration.get_labels_by_name()[0]].selection_state

        validation = []
        for hp_setting, hp_item in selection_state.hp_items.items():
            if hp_setting == selection_state.optimal_hp_setting.get_key():
                for item in hp_item:
                    validation.append(item.performance[metric.name.lower()])

        performances_per_metric[metric.name.lower() if metric.name != "AUC" else metric.name] = {
            "validation": validation,
            "test": [item.performance[metric.name.lower()] for key, item in train_state.optimal_hp_items.items()]
        }

    figure = make_performance_barplot(performances_per_metric, result_path / "validation_vs_test_performance.html")
    figure.show()


def make_performance_barplot(performances_per_metric: dict, result_path: Path = None) -> go.Figure:

    if not result_path.parent.is_dir():
        result_path.mkdir(mode=755)

    fig = go.Figure()

    metrics = list(performances_per_metric.keys())

    for i, group in enumerate(['validation', 'test']):
        if any(len(performances_per_metric[metric][group]) > 1 for metric in metrics):
            fig.add_trace(go.Bar(name=group, x=metrics, y=[np.mean(performances_per_metric[metric][group]) for metric in metrics],
                                 error_y={"type": 'data', 'array': [np.std(performances_per_metric[metric][group]) for metric in metrics]},
                                 marker={'color': px.colors.diverging.Tealrose[i]}))
        else:
            fig.add_trace(go.Bar(name=group, x=metrics, y=[np.mean(performances_per_metric[metric][group]) for metric in metrics],
                                 marker={'color': px.colors.diverging.Tealrose[i]}))

    fig.update_layout(barmode='group', template='plotly_white')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    fig.write_html(result_path)

    return fig
