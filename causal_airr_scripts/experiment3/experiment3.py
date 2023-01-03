import copy
import logging
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReadsType import ReadsType
from sklearn.linear_model import LogisticRegression, RidgeCV, LinearRegression
from sklearn.metrics import confusion_matrix, recall_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from causal_airr_scripts.dataset_util import get_dataset_from_dataframe, write_to_file
from causal_airr_scripts.experiment3.SimConfig import SimConfig, ImplantingSetting, ImplantingGroup
from causal_airr_scripts.experiment3.exp_summary_plots import make_summary, plot_enriched_kmers, merge_dfs, plot_log_reg_coefficients
from causal_airr_scripts.experiment3.sequence_generation import generate_sequences
from causal_airr_scripts.util import overlaps, get_overlap_length


class Experiment3:

    def __init__(self, sim_config: SimConfig, num_processes: int = 4, top_n_coeffs: int = 30):
        self.sim_config = sim_config
        self.num_processes = num_processes
        self.top_n_coeffs = top_n_coeffs

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
        batch_corrected_files = [path / f'batch/with_correction/repetition_{rep}/assessment/enriched_kmer_metrics.tsv'
                                 for rep in range(1, self.sim_config.repetitions + 1)]

        batch_not_corrected_files = [path / f'batch/no_correction/repetition_{rep}/assessment/enriched_kmer_metrics.tsv'
                                     for rep in range(1, self.sim_config.repetitions + 1)]

        control_files = [path / f'control/no_correction/repetition_{rep}/assessment/enriched_kmer_metrics.tsv'
                         for rep in range(1, self.sim_config.repetitions + 1)]

        batch_corrected = merge_dfs(batch_corrected_files, 'repetition', 'batch_corrected')
        batch_baseline = merge_dfs(batch_not_corrected_files, 'repetition', 'batch_baseline')
        control = merge_dfs(control_files, 'repetition', 'control')

        return pd.concat([batch_corrected, batch_baseline, control], axis=0)

    def run_for_impl_setup(self, path: Path, impl_setting: ImplantingSetting, correct: bool):
        setting_path = PathBuilder.build(path / f"{'with_correction' if correct else 'no_correction'}")
        all_metrics = []

        logging.info(f"Starting run for implanting_group: {impl_setting.to_dict()}, correct={correct}")

        for repetition in range(self.sim_config.repetitions):
            metrics = self.run_one_repetition(PathBuilder.build(setting_path / f'repetition_{repetition + 1}'), impl_setting, correct)
            all_metrics.append({**metrics, **{'repetition': repetition + 1}})

        write_to_file(pd.DataFrame(all_metrics), setting_path / 'metrics.tsv')

    def run_one_repetition(self, path: Path, impl_setting: ImplantingSetting, correct: bool) -> dict:
        EnvironmentSettings.set_cache_path(path / 'cache')

        train_dataset = self.simulate_data(PathBuilder.build(path / 'train_dataset'), impl_setting.train, 'train', impl_setting.name)
        test_dataset = self.simulate_data(PathBuilder.build(path / 'test_dataset'), impl_setting.test, 'test', impl_setting.name)

        train_dataset, test_dataset = self.encode_with_kmers(train_dataset, test_dataset, path / 'encoding', correct)

        log_reg = self.train_log_reg(train_dataset)

        metrics = self.assess_log_reg(log_reg, test_dataset, PathBuilder.build(path / 'assessment'))

        shutil.rmtree(path / 'cache')

        return metrics

    def simulate_data(self, path: Path, impl_group: ImplantingGroup, name: str, setting_name: str) -> SequenceDataset:
        df = generate_sequences(self.sim_config.olga_model_name, impl_group, self.sim_config.signal, self.sim_config.p_noise, path, setting_name)
        logging.info(f"Generated {df.shape[0]} sequences for {name} dataset.")
        logging.info(f"Summary:\n\tBatch 0:\n\t\tsignal=True: {df[(df['batch'] == 0) & (df['signal'] == 1)].shape[0]} sequences\n"
                     f"\t\tsignal=False: {df[(df['batch'] == 0) & (df['signal'] == 0)].shape[0]} sequences\n"
                     f"\tBatch 1:\n\t\tsignal=True: {df[(df['batch'] == 1) & (df['signal'] == 1)].shape[0]} sequences\n"
                     f"\t\tsignal=False: {df[(df['batch'] == 1) & (df['signal'] == 0)].shape[0]} sequences\n")

        return get_dataset_from_dataframe(df, path / f'{name}_iml_dataset.tsv', col_mapping=self.col_mapping, meta_col_mapping=self.meta_col_mapping)

    def _assess_kmer_discovery(self, motifs: list, enriched_kmers, k: int):
        if enriched_kmers.shape[0] > 0:
            kmers_in_motifs = sum([int(kmer in [motif.seed for motif in motifs]) for kmer in enriched_kmers]) / enriched_kmers.shape[0]
            kmers_in_partial_motifs = sum([int(any(overlaps(kmer, motif.seed, True) for motif in motifs)) for kmer in enriched_kmers]) / \
                                      enriched_kmers.shape[0]
            recovered_motifs = sum([int(motif.seed in enriched_kmers) for motif in motifs]) / len(motifs)
            recovered_partial_motifs = sum([int(any(overlaps(motif.seed, kmer, True) for kmer in enriched_kmers)) for motif in motifs]) / len(motifs)
        else:
            kmers_in_motifs, kmers_in_partial_motifs, recovered_motifs, recovered_partial_motifs = 0, 0, 0, 0

        kmer_counts_per_overlap = {f"overlap_{i}": [0] for i in range(k + 1)}
        for kmer in enriched_kmers:
            overlap_length = max([get_overlap_length(kmer, motif.seed,
                                                     any(key != 0 and val > 0 for key, val in motif.instantiation._hamming_distance_probabilities.items()))
                                  for motif in motifs])
            kmer_counts_per_overlap[f"overlap_{overlap_length}"][0] += 1

        return {**{"kmers_in_motifs": [kmers_in_motifs], "kmers_in_partial_motifs": [kmers_in_partial_motifs],
                   "recovered_partial_motifs": [recovered_partial_motifs], "recovered_motifs": [recovered_motifs]}, **kmer_counts_per_overlap}

    def encode_with_kmers(self, train_dataset: SequenceDataset, test_dataset: SequenceDataset, path: Path, correct: bool) \
                          -> Tuple[SequenceDataset, SequenceDataset]:

        encoder = KmerFrequencyEncoder.build_object(train_dataset, normalization_type=NormalizationType.RELATIVE_FREQUENCY.name,
                                                    reads=ReadsType.UNIQUE.name, sequence_encoding=self.sim_config.sequence_encoding,
                                                    k=self.sim_config.k, scale_to_unit_variance=False, scale_to_zero_mean=False,
                                                    sequence_type=SequenceType.AMINO_ACID.name)

        encoded_train = encoder.encode(train_dataset, EncoderParams(path, self.label_configuration, learn_model=True))
        encoded_train.encoded_data.labels[self.signal_name] = [1 if val == "True" else 0 for val in
                                                               encoded_train.encoded_data.labels[self.signal_name]]

        encoded_train.encoded_data.examples = encoded_train.encoded_data.examples.todense()
        encoded_train = self.correct_encoded_for_batch(encoded_train, path / 'corrected', correct=correct)

        scaler = StandardScaler()
        encoded_train.encoded_data.examples = scaler.fit_transform(encoded_train.encoded_data.examples)

        encoded_test = encoder.encode(test_dataset, EncoderParams(path, self.label_configuration, learn_model=False))
        encoded_test.encoded_data.labels[self.signal_name] = [1 if val == "True" else 0 for val in encoded_test.encoded_data.labels[self.signal_name]]
        encoded_test.encoded_data.examples = scaler.transform(encoded_test.encoded_data.examples.todense())

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

        baseline_df = pd.DataFrame(train_encoded.examples, columns=[f"baseline_{feature}" for feature in train_encoded.feature_names])
        correction_df = pd.DataFrame(columns=[f"corrected_{feature}" for feature in corrected_train_encoded.feature_names], data=corrected_train_encoded.examples)
        write_to_file(pd.concat([baseline_df, correction_df], axis=1), path / f"train_corrected.tsv")

    def _correct_encoded_dataset(self, dataset: SequenceDataset) -> SequenceDataset:
        corrected_dataset = dataset.clone()
        meta_train_df = dataset.get_metadata([self.signal_name, 'batch'], return_df=True)
        meta_train_df[self.signal_name] = [1 if el == 'True' else 0 for el in meta_train_df['signal']]
        meta_train_df['batch'] = [1 if el == '1' else 0 for el in meta_train_df['batch']]

        for feature_index, feature in enumerate(dataset.encoded_data.feature_names):

            y = dataset.encoded_data.examples[:, feature_index].flatten().tolist()[0]
            lin_reg = LinearRegression(n_jobs=self.num_processes)
            lin_reg.fit(meta_train_df.values, y)

            logging.info(f"Correction model (lin reg) batch coefficient for feature {feature}: {lin_reg.coef_[1]}")

            corrected_y = copy.deepcopy(np.array(y))
            corrected_y[meta_train_df['batch'] == 1] = corrected_y[meta_train_df['batch'] == 1] - lin_reg.coef_[1]
            corrected_dataset.encoded_data.examples[:, feature_index] = corrected_y.reshape(-1, 1)

        return corrected_dataset

    def train_log_reg(self, train_dataset: SequenceDataset) -> LogisticRegression:
        log_reg = LogisticRegression(penalty="l1", solver='saga', max_iter=800)

        clf = GridSearchCV(estimator=log_reg, n_jobs=self.num_processes, param_grid={"C": [1, 10, 100, 1000]}, scoring='balanced_accuracy',
                           cv=3, verbose=0)

        clf.fit(train_dataset.encoded_data.examples, train_dataset.encoded_data.labels[self.signal_name])

        bacc = balanced_accuracy_score(train_dataset.encoded_data.labels[self.signal_name], clf.predict(train_dataset.encoded_data.examples))
        logging.info(f"Training balanced accuracy: {bacc}")

        return clf.best_estimator_

    def assess_log_reg(self, logistic_regression: LogisticRegression, test_dataset: SequenceDataset, path: Path):
        y_pred = logistic_regression.predict(test_dataset.encoded_data.examples)
        true_y = test_dataset.encoded_data.labels[self.signal_name]

        tn, fp, fn, tp = confusion_matrix(true_y, y_pred).ravel()
        labels = self.label_configuration.get_label_values(self.signal_name)

        metrics = {
            'specificity': tn / (tn + fp),
            'sensitivity': recall_score(true_y, y_pred, labels=labels),
            'balanced_accuracy': balanced_accuracy_score(true_y, y_pred),
            'auc': roc_auc_score(true_y, y_pred, labels=labels)
        }

        self._save_metrics(metrics, path)
        plot_log_reg_coefficients(logistic_regression, test_dataset.encoded_data.feature_names, self.top_n_coeffs, self.sim_config.signal.motifs, path)

        sorted_indices = np.argsort(np.abs(logistic_regression.coef_.flatten()))[-self.top_n_coeffs:]
        enriched_kmers = np.array(test_dataset.encoded_data.feature_names)[sorted_indices]
        kmer_df = pd.DataFrame(self._assess_kmer_discovery(self.sim_config.signal.motifs, enriched_kmers, self.sim_config.k))
        write_to_file(kmer_df, path / 'enriched_kmer_metrics.tsv')

        return metrics

    def _save_metrics(self, metrics: dict, path: Path):
        with open(path / 'metrics.yaml', 'w') as file:
            yaml.dump(metrics, file)
