from pathlib import Path

import yaml

from causal_airr_scripts.plotting import plot_multiple_boxplots


def main():
    """
    The code to make manuscript figure 1b merging all the confounder distributions in one plot
    """
    datasets = {}
    for path, name in zip(
        [Path("results_exp1/2023-03-01_15:54:06.008778/1a/balanced_error_rate_performances.yaml"),
         Path("results_exp1/2023-03-01_15:54:06.008778/1b/balanced_error_rate_performances.yaml"),
         Path("results_exp1/2023-03-01_15:54:06.008778/1c/balanced_error_rate_performances.yaml")],
        ['stable confounder', 'minor changes in confounder', 'major changes in confounder']):
        with path.open('r') as file:
            datasets[name] = yaml.safe_load(file)

    plot_multiple_boxplots(datasets, result_path="./exp1.html", decimal_count=2)


if __name__ == '__main__':
    main()
