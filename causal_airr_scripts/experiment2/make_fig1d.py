from pathlib import Path

import yaml

from causal_airr_scripts.plotting import plot_multiple_boxplots


def main():
    """
    The code to make manuscript figure 1d showing what happens with selection bias
    """
    datasets = {}
    for path, name in zip(
        [Path("../../results_exp2/2023-03-02_17:41:49.793863/2a/balanced_error_rate_performances.yaml"),
         Path("../../results_exp2/2023-03-02_17:41:49.793863/2b/balanced_error_rate_performances.yaml")],
        ['selection bias in validation, but not in test', 'only spurious correlation in validation (no immune signal)']):
        with path.open('r') as file:
            datasets[name] = yaml.safe_load(file)

    plot_multiple_boxplots(datasets, result_path="./exp2.html")


if __name__ == '__main__':
    main()
