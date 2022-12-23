import copy
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml
from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReadsType import ReadsType
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, recall_score, balanced_accuracy_score
from statsmodels.stats.multitest import fdrcorrection

from causal_airr_scripts.dataset_util import get_dataset_from_dataframe, write_to_file
from causal_airr_scripts.experiment3.SimConfig import SimConfig, ImplantingGroup, ImplantingUnit
from causal_airr_scripts.experiment3.exp_summary_plots import make_summary, plot_enriched_kmers, merge_dfs, plot_log_reg_coefficients
from causal_airr_scripts.experiment3.sequence_generation import generate_sequences
from causal_airr_scripts.kmer_enrichment import make_contingency_table, compute_p_value
from causal_airr_scripts.util import overlaps, get_overlap_length


class Experiment3:

    def __init__(self, sim_config: SimConfig, p_correction_points: float, test_percentage: float = 0.3):
        self.sim_config = sim_config
        self.p_correction_points = p_correction_points
        self.test_percentage = test_percentage

    @property
    def label_configuration(self) -> LabelConfiguration:
        return LabelConfiguration(labels=[Label(self.signal_name, [True, False])])

    @property
    def signal_name(self) -> str:
        return self.sim_config.signal.id

    @property
    def col_mapping(self) -> dict:
        return {0: 'sequence_aas', 1: 'v_genes', 2: 'j_genes', 3: self.signal_name, 4: 'batch'}

    @property
    def meta_col_mapping(self) -> dict:
        return {self.signal_name: self.signal_name, 'batch': 'batch'}

    def run(self, path: Path):

        self.run_for_impl_setup(PathBuilder.build(path / 'control'), self.sim_config.implanting_config.control, correct=False)

        for correct in [True, False]:
            self.run_for_impl_setup(PathBuilder.build(path / 'batch'), self.sim_config.implanting_config.batch, correct)

        self.make_reports(path)

    def make_reports(self, path: Path):
        make_summary(path / "batch/with_correction/metrics.tsv", path / "batch/no_correction/metrics.tsv", path / "control/no_correction/metrics.tsv",
                     path)

        enriched_kmer_df = self._make_enriched_kmer_summary_df(path)
        plot_enriched_kmers(path, enriched_kmer_df, self.sim_config.k)

    def _make_enriched_kmer_summary_df(self, path: Path) -> pd.DataFrame:
        batch_corrected_files = [path / f'batch/with_correction/repetition_{rep}/enriched_k-mers/metrics.tsv'
                                 for rep in range(1, self.sim_config.repetitions + 1)]

        batch_not_corrected_files = [path / f'batch/no_correction/repetition_{rep}/enriched_k-mers/metrics.tsv'
                                     for rep in range(1, self.sim_config.repetitions + 1)]

        control_files = [path / f'control/no_correction/repetition_{rep}/enriched_k-mers/metrics.tsv'
                         for rep in range(1, self.sim_config.repetitions + 1)]

        batch_corrected = merge_dfs(batch_corrected_files, 'repetition', 'batch_corrected')
        batch_baseline = merge_dfs(batch_not_corrected_files, 'repetition', 'batch_baseline')
        control = merge_dfs(control_files, 'repetition', 'control')

        return pd.concat([batch_corrected, batch_baseline, control], axis=0)

    def run_for_impl_setup(self, path: Path, impl_group: ImplantingGroup, correct: bool):
        setting_path = PathBuilder.build(path / f"{'with_correction' if correct else 'no_correction'}")
        all_metrics = []

        logging.info(f"Starting run for implanting_group: {impl_group.to_dict()}, correct={correct}")

        for repetition in range(self.sim_config.repetitions):
            metrics = self.run_one_repetition(PathBuilder.build(setting_path / f'repetition_{repetition + 1}'), impl_group, correct)
            all_metrics.append({**metrics, **{'repetition': repetition + 1}})

        write_to_file(pd.DataFrame(all_metrics), setting_path / 'metrics.tsv')

    def run_one_repetition(self, path: Path, impl_group: ImplantingGroup, correct: bool) -> dict:
        train_dataset = self.simulate_data(PathBuilder.build(path / 'dataset'), impl_group)

        self.report_enriched_kmers(train_dataset, PathBuilder.build(path / f'enriched_k-mers'))

        train_dataset = get_dataset_from_dataframe(train_dataset, path / 'iml_dataset.tsv', col_mapping=self.col_mapping,
                                                   meta_col_mapping=self.meta_col_mapping)

        test_dataset = self.make_test_dataset(PathBuilder.build(path / 'test_dataset'), train_dataset.get_example_count())

        train_dataset, test_dataset = self.encode_with_kmers(train_dataset, test_dataset, path / 'encoding')
        train_dataset = self.correct_encoded_for_batch(train_dataset, path / 'correction', correct)
        log_reg = self.train_log_reg(train_dataset)

        metrics = self.assess_log_reg(log_reg, test_dataset, PathBuilder.build(path / 'assessment'))
        return metrics

    def simulate_data(self, path: Path, impl_group: ImplantingGroup) -> pd.DataFrame:
        df = generate_sequences(self.sim_config.olga_model_name, impl_group, self.sim_config.signal, self.sim_config.p_noise, path)
        logging.info(f"Generated {df.shape[0]} sequences.")
        logging.info(f"Summary:\n\tBatch 0:\n\t\tsignal=True: {df[(df['batch'] == 0) & (df['signal'] == 1)].shape[0]} sequences\n"
                     f"\t\tsignal=False: {df[(df['batch'] == 0) & (df['signal'] == 0)].shape[0]} sequences\n"
                     f"\tBatch 1:\n\t\tsignal=True: {df[(df['batch'] == 1) & (df['signal'] == 1)].shape[0]} sequences\n"
                     f"\t\tsignal=False: {df[(df['batch'] == 1) & (df['signal'] == 0)].shape[0]} sequences\n")

        return df

    def report_enriched_kmers(self, dataset: pd.DataFrame, path: Path, repetition_index: int = 1):
        contingency_table = make_contingency_table(dataset, self.sim_config.k)
        contingency_with_p_value = compute_p_value(contingency_table)
        write_to_file(contingency_with_p_value, path / f'all_kmers_with_p_value.tsv')

        motifs = self.sim_config.signal.motifs
        results = []

        for fdr in self.sim_config.fdrs:
            kmer_selection, contingency_with_p_value['q_value'] = fdrcorrection(contingency_with_p_value.p_value, alpha=fdr)
            enriched_kmers = contingency_with_p_value[kmer_selection]

            write_to_file(enriched_kmers, path / f"enriched_kmers_{fdr}.tsv")

            metrics = self._assess_kmer_discovery(motifs, enriched_kmers, self.sim_config.k)
            results.append({**metrics, **{'FDR': fdr}})

        write_to_file(pd.DataFrame(results), path / 'metrics.tsv')

    def _assess_kmer_discovery(self, motifs: list, enriched_kmers, k: int):
        if enriched_kmers.shape[0] > 0:
            kmers_in_motifs = sum([int(kmer in [motif.seed for motif in motifs]) for kmer in enriched_kmers.index]) / enriched_kmers.shape[0]
            kmers_in_partial_motifs = sum([int(any(overlaps(kmer, motif.seed, True) for motif in motifs)) for kmer in enriched_kmers.index]) / \
                                      enriched_kmers.shape[0]
            recovered_motifs = sum([int(motif.seed in enriched_kmers.index) for motif in motifs]) / len(motifs)
            recovered_partial_motifs = sum([int(any(overlaps(motif.seed, kmer, True) for kmer in enriched_kmers.index)) for motif in motifs]) / len(motifs)
        else:
            kmers_in_motifs, kmers_in_partial_motifs, recovered_motifs, recovered_partial_motifs = 0, 0, 0, 0

        kmer_counts_per_overlap = {f"overlap_{i}": 0 for i in range(k + 1)}
        for kmer in enriched_kmers.index:
            overlap_length = max([get_overlap_length(kmer, motif.seed,
                                                     any(key != 0 and val > 0 for key, val in motif.instantiation._hamming_distance_probabilities.items()))
                                  for motif in motifs])
            kmer_counts_per_overlap[f"overlap_{overlap_length}"] += 1

        return {**{"kmers_in_motifs": kmers_in_motifs, "kmers_in_partial_motifs": kmers_in_partial_motifs,
                   "recovered_partial_motifs": recovered_partial_motifs, "recovered_motifs": recovered_motifs}, **kmer_counts_per_overlap}

    def make_test_dataset(self, path: Path, train_data_size: int) -> SequenceDataset:
        test_dataset = self.simulate_data(path, ImplantingGroup(baseline=ImplantingUnit(0.5, []), modified=ImplantingUnit(0.5, []),
                                                                seq_count=round(self.test_percentage * train_data_size), name='test'))

        test_dataset = get_dataset_from_dataframe(test_dataset, path / 'test_dataset.tsv', self.col_mapping, self.meta_col_mapping)

        return test_dataset

    def encode_with_kmers(self, train_dataset: SequenceDataset, test_dataset: SequenceDataset, path: Path) -> Tuple[SequenceDataset, SequenceDataset]:

        encoder = KmerFrequencyEncoder.build_object(train_dataset, normalization_type=NormalizationType.RELATIVE_FREQUENCY.name,
                                                    reads=ReadsType.UNIQUE.name, sequence_encoding=self.sim_config.sequence_encoding,
                                                    k=self.sim_config.k, scale_to_unit_variance=True, scale_to_zero_mean=True,
                                                    sequence_type=SequenceType.AMINO_ACID.name)

        encoded_train = encoder.encode(train_dataset, EncoderParams(path, self.label_configuration, learn_model=True))
        encoded_train.encoded_data.labels[self.signal_name] = [1 if val == "True" else 0 for val in
                                                               encoded_train.encoded_data.labels[self.signal_name]]
        encoded_test = encoder.encode(test_dataset, EncoderParams(path, self.label_configuration, learn_model=False))
        encoded_test.encoded_data.labels[self.signal_name] = [1 if val == "True" else 0 for val in encoded_test.encoded_data.labels[self.signal_name]]

        return encoded_train, encoded_test

    def correct_encoded_for_batch(self, train_dataset: SequenceDataset, path: Path, correct: bool) -> SequenceDataset:

        if correct:
            PathBuilder.build(path)

            corrected_train = self._correct_encoded_dataset(train_dataset)
            self._report_correction_diff(train_dataset.encoded_data, corrected_train.encoded_data, path)

            return corrected_train
        else:
            return train_dataset

    def _report_correction_diff(self, train_encoded: EncodedData, corrected_train_encoded: EncodedData, path: Path):

        correction_df = pd.DataFrame(train_encoded.examples, columns=[f"baseline_{feature}" for feature in train_encoded.feature_names])
        correction_df[[f"corrected_{feature}" for feature in corrected_train_encoded.feature_names]] = corrected_train_encoded.examples
        write_to_file(correction_df, path / f"train_corrected.tsv")

    def _correct_encoded_dataset(self, dataset: SequenceDataset) -> SequenceDataset:
        corrected_dataset = dataset.clone()
        meta_train_df = dataset.get_metadata([self.signal_name, 'batch'], return_df=True)
        meta_train_df[self.signal_name] = [1 if el == 'True' else 0 for el in meta_train_df['signal']]
        meta_train_df['batch'] = [1 if el == '1' else 0 for el in meta_train_df['batch']]
        for feature_index, feature in enumerate(dataset.encoded_data.feature_names):
            y = dataset.encoded_data.examples[:, feature_index]
            lin_reg = LinearRegression(n_jobs=4)
            lin_reg.fit(meta_train_df.values, y)
            logging.info(f"Correction model (lin reg) batch coefficient for feature {feature}: {lin_reg.coef_[1]}")
            corrected_y = copy.deepcopy(y)
            corrected_y[meta_train_df['batch'] == 1] = corrected_y[meta_train_df['batch'] == 1] - lin_reg.coef_[1]
            corrected_dataset.encoded_data.examples[:, feature_index] = corrected_y

        return corrected_dataset

    def train_log_reg(self, train_dataset: SequenceDataset) -> LogisticRegression:
        log_reg = LogisticRegression(penalty="l1", solver='saga', n_jobs=4, C=100)
        log_reg.fit(train_dataset.encoded_data.examples, train_dataset.encoded_data.labels[self.signal_name])

        bacc = balanced_accuracy_score(train_dataset.encoded_data.labels[self.signal_name], log_reg.predict(train_dataset.encoded_data.examples))
        logging.info(f"Training balanced accuracy: {bacc}")

        return log_reg

    def assess_log_reg(self, logistic_regression: LogisticRegression, test_dataset: SequenceDataset, path: Path):
        y_pred = logistic_regression.predict(test_dataset.encoded_data.examples)
        true_y = test_dataset.encoded_data.labels[self.signal_name]
        self._save_predictions(y_pred, true_y, path)

        tn, fp, fn, tp = confusion_matrix(true_y, y_pred).ravel()
        labels = self.label_configuration.get_label_values(self.signal_name)

        metrics = {
            'specificity': tn / (tn + fp),
            'sensitivity': recall_score(true_y, y_pred, labels=labels),
            'balanced_accuracy': balanced_accuracy_score(true_y, y_pred),
        }

        self._save_metrics(metrics, path)
        plot_log_reg_coefficients(logistic_regression, test_dataset.encoded_data.feature_names, 20, self.sim_config.signal.motifs, path)

        return metrics

    def _save_predictions(self, y_pred, true_y, path: Path):
        write_to_file(pd.DataFrame({"predicted_label": y_pred, "true_label": true_y}), path / 'predictions.tsv')

    def _save_metrics(self, metrics: dict, path: Path):
        with open(path / 'metrics.yaml', 'w') as file:
            yaml.dump(metrics, file)
