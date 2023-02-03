import argparse
from pathlib import Path

import yaml

from causal_airr_scripts.experiment3.SimConfig import SimConfig
from causal_airr_scripts.experiment3.experiment3 import Experiment3


def prepare_namespace():
    parser = argparse.ArgumentParser(description="CausalAIRR experiment 3")
    parser.add_argument("specification_path", help="Path to specification YAML file. Always used to define the analysis.")
    parser.add_argument("result_path", help="Output directory path.")

    namespace = parser.parse_args()
    namespace.specification_path = Path(namespace.specification_path)
    namespace.result_path = Path(namespace.result_path)
    return namespace


def main():
    namespace = prepare_namespace()
    with open(namespace.specification_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    config = SimConfig.from_dict(config_dict)

    experiment = Experiment3(config, 0.01)
    experiment.run(namespace.result_path)


if __name__ == "__main__":
    main()
