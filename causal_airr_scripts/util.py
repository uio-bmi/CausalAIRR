import copy

import yaml

from causal_airr_scripts.experiment3.SimConfig import SimConfig


def overlaps(word1, word2) -> bool:
    return bool(get_overlap_length(word1, word2))


def get_overlap_length(word1, word2) -> int:
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

    k = len(min_word) - 1

    while k >= 1 and not overlap:

        if max_word.startswith(min_word[k:]) or max_word.endswith(min_word[:k]) or \
           min_word.startswith(max_word[-k:]) or min_word.endswith(max_word[:k]):

            overlap = True
            overlap_length = k
        k -= 1

    return overlap_length


def write_config(config: SimConfig, path):

    all_config = config.to_dict()

    with open(path / 'config.yaml', 'w') as file:
        yaml.dump(all_config, file)
