import argparse
from pathlib import Path

import yaml

from causal_airr_scripts.experiment3.SimConfig import SimConfig


def overlaps(word1, word2, middle_replacement_allowed: bool = False) -> bool:
    return bool(get_overlap_length(word1, word2, middle_replacement_allowed))


def get_overlap_length(word1, word2, middle_replacement_allowed: bool) -> int:
    if len(word1) > len(word2):
        max_word, min_word = word1, word2
    else:
        max_word, min_word = word2, word1

    if min_word in max_word[:len(min_word)] or min_word in max_word[-len(min_word):]:
        overlap = True
        overlap_length = len(min_word)
    else:
        overlap = False
        overlap_length = 0

    if middle_replacement_allowed and not overlap and len(min_word) == len(max_word):
        if max_word[0] == min_word[0] and max_word[-1] == min_word[-1]:
            overlap_length = 2
            overlap = True

    k = len(min_word) - 1

    while k >= 1 and not overlap:

        if max_word.startswith(min_word[-k:]) or max_word.endswith(min_word[:k]) or \
           min_word.startswith(max_word[-k:]) or min_word.endswith(max_word[:k]):

            overlap = True
            overlap_length = k
        k -= 1

    return overlap_length


def write_config(config: SimConfig, path):

    all_config = config.to_dict()

    with open(path / 'config.yaml', 'w') as file:
        yaml.dump(all_config, file)


def save_to_yaml(content: dict, path: Path):
    with path.open('w') as file:
        yaml.dump(content, file)


def prepare_namespace():
    parser = argparse.ArgumentParser(description="CausalAIRR experiments")
    parser.add_argument("result_path", help="Output directory path.")
    parser.add_argument("num_processes", help="Number of processes to use for training logistic regression.", type=int)

    namespace = parser.parse_args()
    namespace.result_path = Path(namespace.result_path)
    return namespace