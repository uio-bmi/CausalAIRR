import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from immuneML.ml_metrics.Metric import Metric


def plot_balanced_error_rate(iml_result: list, result_path):
    train_state = iml_result[0]

    assert Metric['BALANCED_ACCURACY'] in train_state.metrics or Metric['BALANCED_ACCURACY'] == train_state.optimization_metric

    selection_state = train_state.assessment_states[0].label_states[train_state.label_configuration.get_labels_by_name()[0]].selection_state

    validation = []
    for hp_setting, hp_item in selection_state.hp_items.items():
        if hp_setting == selection_state.optimal_hp_setting.get_key():
            for item in hp_item:
                validation.append(1 - item.performance['balanced_accuracy'])

    performances = {
        "validation": validation,
        "test": [1 - train_state.optimal_hp_items['immune_state'].performance['balanced_accuracy']]
    }

    figure = plot_error_rate_box(performances, result_path / "validation_vs_test_performance_balanced_error_rate.html")
    figure.show()


def plot_error_rate_box(data: dict, result_path):

    figure = go.Figure()
    decimal_count = 3

    i = 0
    for key in data:
        figure.add_box(boxpoints='all', name=key, y=data[key], marker_color=px.colors.diverging.Tealrose[i], text=np.median(data[key]).round(decimal_count))
        figure.add_annotation(x=key, y=np.median(data[key]), text=str(np.median(data[key]).astype(float).round(decimal_count)), showarrow=False, yshift=15, font_color='black')
        i += 1

    figure.update_layout(xaxis={"title": "balanced error rates", 'color': 'black'}, template='plotly_white')
    figure.update_xaxes(showline=True, linewidth=1, linecolor='black')
    figure.update_yaxes(showline=True, linewidth=1, linecolor='black', color='black')

    figure.write_html(result_path)
    return figure
