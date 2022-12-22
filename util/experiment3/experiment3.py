import copy
import random
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
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.hyperparameter_optimization.config.SplitConfig import SplitConfig
from immuneML.hyperparameter_optimization.config.SplitType import SplitType
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReadsType import ReadsType
from immuneML.workflows.steps.data_splitter.DataSplitter import DataSplitter
from immuneML.workflows.steps.data_splitter.DataSplitterParams import DataSplitterParams
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, recall_score, balanced_accuracy_score, roc_auc_score
from statsmodels.stats.multitest import fdrcorrection
import plotly.graph_objects as go

from AIRR_experiment_3 import generate_sequences, assess_performance
from util.SimConfig import SimConfig
from util.dataset_util import write_to_file
from util.experiment3.exp_summary import make_summary
from util.exploration import get_dataset_from_dataframe
from util.kmer_enrichment import make_contingency_table, compute_p_value


class Experiment3:

    def __init__(self, sim_config: SimConfig, p_correction_points: float):
        self.sim_config = sim_config
        self.p_correction_points = p_correction_points

    @property
    def label_configuration(self) -> LabelConfiguration:
        return LabelConfiguration(labels=[Label(self.signal_name, [True, False])])

    @property
    def signal_name(self) -> str:
        return self.sim_config.signal.id

    def run(self, path: Path):

        for correct in [True, False]:
            setting_path = PathBuilder.build(path / f"{'with_correction' if correct else 'no_correction'}")
            all_metrics = []
            for repetition in range(self.sim_config.repetitions):
                metrics = self.run_one_repetition(PathBuilder.build(setting_path / f'repetition_{repetition+1}'), correct)
                all_metrics.append({**metrics, **{'repetition': repetition + 1}})

            write_to_file(pd.DataFrame(all_metrics), setting_path / 'metrics.tsv')

        make_summary(path / "with_correction/metrics.tsv", path / "no_correction/metrics.tsv", path)

    def run_one_repetition(self, path: Path, correct: bool) -> dict:
        dataset = self.simulate_data(PathBuilder.build(path / 'dataset'))

        self.report_enriched_kmers(dataset, PathBuilder.build(path / f'enriched_{self.sim_config.k}-mers'))

        dataset = get_dataset_from_dataframe(dataset, path / 'iml_dataset.tsv',
                                             col_mapping={0: 'sequence_aas', 1: 'v_genes', 2: 'j_genes', 3: self.signal_name, 4: 'batch'},
                                             meta_col_mapping={self.signal_name: self.signal_name, 'batch': 'batch'})

        train_dataset, test_dataset = self.split_to_train_test(dataset, PathBuilder.build(path / 'split_datasets'))
        train_dataset, test_dataset = self.encode_with_kmers(train_dataset, test_dataset, PathBuilder.build(path / "encoding"))
        train_dataset, test_dataset = self.correct_encoded_for_batch(train_dataset, test_dataset, PathBuilder.build(path / 'correction'), correct)
        log_reg = self.train_log_reg(train_dataset)

        metrics = self.assess_log_reg(log_reg, test_dataset, PathBuilder.build(path / 'assessment'))
        return metrics

    def simulate_data(self, path: Path) -> pd.DataFrame:
        if self.sim_config.implanting_config.control is not None:
            raise NotImplementedError

        return generate_sequences(self.sim_config.olga_model_name, self.sim_config.implanting_config.batch, self.sim_config.signal,
                                  self.sim_config.p_noise, path)

    def report_enriched_kmers(self, dataset: pd.DataFrame, path: Path, repetition_index: int = 1):
        contingency_table = make_contingency_table(dataset, self.sim_config.k)
        contingency_with_p_value = compute_p_value(contingency_table)
        write_to_file(contingency_with_p_value, path / f'all_{self.sim_config.k}mers_with_p_value.tsv')

        motifs = self.sim_config.signal.motifs
        results = []

        for fdr in self.sim_config.fdrs:
            kmer_selection, contingency_with_p_value['q_value'] = fdrcorrection(contingency_with_p_value.p_value, alpha=fdr)
            enriched_kmers = contingency_with_p_value[kmer_selection]

            write_to_file(enriched_kmers, path / f"enriched_{self.sim_config.k}mers_{fdr}.tsv")

            metrics = assess_performance(motifs, enriched_kmers, self.sim_config.k)
            results.append({**metrics, **{'FDR': fdr}})

        write_to_file(pd.DataFrame(results), path / 'metrics.tsv')

    def split_to_train_test(self, dataset: SequenceDataset, path: Path) -> Tuple[SequenceDataset, SequenceDataset]:
        train, test = DataSplitter.random_split(DataSplitterParams(dataset, split_count=1, training_percentage=0.7, paths=[path],
                                                                   split_strategy=SplitType.RANDOM, label_config=self.label_configuration,
                                                                   split_config=SplitConfig(split_strategy=SplitType.RANDOM, split_count=1,
                                                                                            training_percentage=0.7)))

        return train[0], test[0]

    def encode_with_kmers(self, train_dataset: SequenceDataset, test_dataset: SequenceDataset, path: Path) -> Tuple[SequenceDataset, SequenceDataset]:

        encoder = KmerFrequencyEncoder.build_object(train_dataset, normalization_type=NormalizationType.RELATIVE_FREQUENCY.name,
                                                    reads=ReadsType.UNIQUE.name, sequence_encoding=self.sim_config.sequence_encoding,
                                                    k=self.sim_config.k, scale_to_unit_variance=True, scale_to_zero_mean=True,
                                                    sequence_type=SequenceType.AMINO_ACID.name)

        encoded_train = encoder.encode(train_dataset, EncoderParams(path, self.label_configuration, learn_model=True))
        encoded_train.encoded_data.labels[self.signal_name] = [1 if val == "True" else 0 for val in encoded_train.encoded_data.labels[self.signal_name]]
        encoded_test = encoder.encode(test_dataset, EncoderParams(path, self.label_configuration, learn_model=False))
        encoded_test.encoded_data.labels[self.signal_name] = [1 if val == "True" else 0 for val in encoded_test.encoded_data.labels[self.signal_name]]

        print(f"Finished encoding...")

        return encoded_train, encoded_test

    def correct_encoded_for_batch(self, train_dataset: SequenceDataset, test_dataset: SequenceDataset, path: Path, correct: bool) \
                                  -> Tuple[SequenceDataset, SequenceDataset]:

        if correct:
            corrected_train, lin_reg = self._correct_encoded_dataset(train_dataset)
            corrected_test, _ = self._correct_encoded_dataset(test_dataset, lin_reg)

            self._report_correction_diff(train_dataset.encoded_data, corrected_train.encoded_data, test_dataset.encoded_data,
                                         corrected_test.encoded_data, path)

            return corrected_train, corrected_test
        else:
            return train_dataset, test_dataset

    def _report_correction_diff(self, train_encoded: EncodedData, corrected_train_encoded: EncodedData, test_encoded: EncodedData,
                                corrected_test_encoded: EncodedData, path: Path):

        for baseline, corrected in [[train_encoded, corrected_train_encoded], [test_encoded, corrected_test_encoded]]:

            figure = go.Figure()

            for feature_index, feature in enumerate(baseline.feature_names):

                if random.uniform(0, 1) < self.p_correction_points:

                    y = np.concatenate([baseline.examples[:, feature_index], corrected.examples[:, feature_index]], axis=0)
                    x = [f"baseline_{feature}" for _ in range(baseline.examples.shape[0])] + \
                        [f'corrected_{feature}' for _ in range(corrected.examples.shape[0])]

                    figure.add_trace(go.Box(name=feature, x=x, y=y, opacity=0.7, offsetgroup=feature, marker={'opacity': 0.5},
                                            legendgroup=feature))

            correction_df = pd.DataFrame(baseline.examples, columns=[f"baseline_{feature}" for feature in baseline.feature_names])
            correction_df[[f"corrected_{feature}" for feature in corrected.feature_names]] = corrected.examples
            write_to_file(correction_df, path / f"{'train' if baseline == train_encoded else 'test'}_corrected.tsv")

            file_path = path / f"subset_{'train' if baseline == train_encoded else 'test'}_corrected.html"
            figure.write_html(str(file_path))

    def _correct_encoded_dataset(self, dataset: SequenceDataset, lin_reg: LinearRegression = None) -> Tuple[SequenceDataset, LinearRegression]:
        corrected_dataset = dataset.clone()
        meta_train_df = dataset.get_metadata([self.signal_name, 'batch'], return_df=True)
        meta_train_df['signal'] = [1 if el == 'True' else 0 for el in meta_train_df['signal']]
        meta_train_df['batch'] = [1 if el == '1' else 0 for el in meta_train_df['batch']]
        for feature_index, feature in enumerate(dataset.encoded_data.feature_names):
            y = dataset.encoded_data.examples[:, feature_index]
            if lin_reg is None:
                lin_reg = LinearRegression(n_jobs=4)
                lin_reg.fit(meta_train_df.values, y)
            corrected_y = copy.deepcopy(y)
            corrected_y[meta_train_df['batch'] == 1] = corrected_y[meta_train_df['batch'] == 1] - lin_reg.coef_[1]
            corrected_dataset.encoded_data.examples[:, feature_index] = corrected_y

        return corrected_dataset, lin_reg

    def train_log_reg(self, train_dataset: SequenceDataset) -> LogisticRegression:
        log_reg = LogisticRegression(penalty="l1", solver='saga', n_jobs=4, C=100)
        log_reg.fit(train_dataset.encoded_data.examples, train_dataset.encoded_data.labels[self.signal_name])

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
            'auc': roc_auc_score(true_y, y_pred, labels=labels)
        }

        self._save_metrics(metrics, path)
        print(metrics)

        return metrics

    def _save_predictions(self, y_pred, true_y, path: Path):
        write_to_file(pd.DataFrame({"predicted_label": y_pred, "true_label": true_y}), path / 'predictions.tsv')

    def _save_metrics(self, metrics: dict, path: Path):
        with open(path / 'metrics.yaml', 'w') as file:
            yaml.dump(metrics, file)
