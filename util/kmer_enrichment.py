import itertools

import fisher
import numpy as np
import pandas as pd
from immuneML.util.KmerHelper import KmerHelper


def compute_contingency_row(kmers_per_sequence, sequences, kmer):
    present_in_positive = np.sum(kmers_per_sequence[kmer][np.logical_and(kmers_per_sequence[kmer].values, sequences['signal'].values)].values)
    present_in_negative = np.sum(
        kmers_per_sequence[kmer][np.logical_and(kmers_per_sequence[kmer].values, np.logical_not(sequences['signal'].values))].values)
    absent_in_positive = np.sum(np.logical_and(sequences['signal'].values, kmers_per_sequence[kmer].values == 0))
    absent_in_negative = np.sum(np.logical_and(np.logical_not(sequences['signal'].values), kmers_per_sequence[kmer].values == 0))
    row = (present_in_positive, present_in_negative, absent_in_positive, absent_in_negative)
    assert np.sum(row) == sequences.shape[0], (np.sum(row), sequences.shape[0])
    return row


def compute_contingency_values(kmers_per_sequence, sequences):
    contingency_table = np.zeros(shape=(kmers_per_sequence.columns.shape[0], 4), dtype=int)

    for index, kmer in enumerate(list(kmers_per_sequence.columns)):
        contingency_table[index] = compute_contingency_row(kmers_per_sequence, sequences, kmer)

    return contingency_table


def make_contingency_table(sequences: pd.DataFrame, k: int) -> pd.DataFrame:
    kmers_per_sequence = get_kmer_presence_from_sequences(sequences, k)

    contingency_table = compute_contingency_values(kmers_per_sequence, sequences)
    contingency_table_df = pd.DataFrame(contingency_table, index=list(kmers_per_sequence.columns),
                                        columns=['present_in_positive', 'present_in_negative', 'absent_in_positive', 'absent_in_negative'])

    contingency_table_df['kmer'] = contingency_table_df.index

    return contingency_table_df


def find_enriched_kmers(contingency_table: pd.DataFrame, fdr: float):
    relevant_kmers = contingency_table[contingency_table['q_value'] < fdr]
    return relevant_kmers[['kmer', 'q_value']]


def compute_p_value(contingency_table: pd.DataFrame):
    contingency_table['p_value'] = contingency_table.apply(lambda row: fisher.pvalue(row['present_in_positive'], row['present_in_negative'],
                                                                                     row['absent_in_positive'], row['absent_in_negative']).right_tail,
                                                           axis=1)

    return contingency_table


def get_kmer_presence_from_sequences(sequences: pd.DataFrame, k: int):
    kmers = [KmerHelper.create_kmers_from_string(seq, k) for seq in sequences['sequence_aa']]
    unique_kmers = list(set(itertools.chain.from_iterable(kmers)))
    kmer_per_sequence = pd.DataFrame(data=np.zeros((sequences.shape[0], len(unique_kmers)), dtype=int), columns=unique_kmers)
    for index, kmer_list in enumerate(kmers):
        unique_list = list(set(kmer_list))
        kmer_per_sequence.at[index, unique_list] = 1

    return kmer_per_sequence


def compute_q_values(p_values: np.ndarray, pi0: float = 0.999) -> np.ndarray:
    sorted_indices = np.argsort(p_values)
    m = float(p_values.shape[0])
    q_values = pi0 * p_values[sorted_indices]
    q_values[-1] = min(q_values[-1], 1.0)
    for i in range(int(m) - 2, -1, -1):
        q_values[i] = min(q_values[i] * m / (i+1.),  # i+1 because it's zero based
                          q_values[i+1])
        print(f"{i}-th element: {q_values[i]}")

    return q_values[sorted_indices]


def make_seqs_with_permuted_labels(sequences: pd.DataFrame) -> pd.DataFrame:
    permuted_seqs = sequences.copy(deep=True)
    permuted_seqs['signal'] = np.random.permutation(permuted_seqs['signal'])
    return permuted_seqs
