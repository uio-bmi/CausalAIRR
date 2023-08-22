import copy
import logging
import re
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.implants.Signal import Signal
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReadsType import ReadsType
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, recall_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from causal_airr_scripts.dataset_util import get_dataset_from_dataframe, write_to_file
from causal_airr_scripts.experiment3.SimConfig import SimConfig, ImplantingSetting, ImplantingGroup, make_signal
from causal_airr_scripts.experiment3.exp_summary_plots import make_summary, plot_enriched_kmers, merge_dfs, plot_log_reg_coefficients
from causal_airr_scripts.experiment3.sequence_generation import generate_sequences
from causal_airr_scripts.util import overlaps, get_overlap_length, save_to_yaml


class Experiment3:
    """
    Experiment3 class encapsulates experimental runs illustrating the influence of the presence of batch effects on
    predicting adaptive immune receptor specificity. It uses SimConfig object which contains all simulation parameters.

    The simulation is defined as follows:
    - some parameters are general and apply to the whole simulation, such as k-mer length, which model to use to
    generate background sequences (before any motifs/signals are added), how many repetitions of the whole simulation
    to make, which specific batch correction approaches to try, etc.
    - some parameters define how to simulate the data in more detail - this is what ImplantingConfig class defines

    The data are simulated in two groups (ImplantingConfig class):
    1. control receptors which only have the immune signal
    2. receptors with both batch effects and the immune signal

    This second group of receptors consists of 50% receptors from batch 0 (no batch signal added) and 50% of receptors
    from batch 1 (with batch signal added) (defined in ImplantingGroup class). The batch effect signals and immune
    signals may be correlated as defined in ImplantingUnit class.

    Additionally, we allow that these correlations differ between training and test to illustrate what happens with
    batch effects and ML models when they are transferred to a new setting (or same setting with different
    distributions).

    After the simulation, in the analysis there are 3 groups of receptor sequences which are used to compare ML
    performance:
    1. control receptors (only have the immune signal, no batch effects - ideal case),
    2. receptors that have batch effects which are corrected for (it could be more of them, but in the analysis we
    performed there was only one way of correcting: LinearRegression),
    3. receptors that have batch effects which are not corrected for

    """
    def __init__(self, sim_config: SimConfig, num_processes: int = 4, top_n_coeffs: int = 30):
        """
        Arguments:
            sim_config (SimConfig): parameters of the simulation
            num_processes (int): how many simulations to run in parallel
            top_n_coeffs (int): for final ML models (logistic regression), how many top coefficients to show in plot
        """
        self.sim_config = sim_config
        self.num_processes = num_processes
        self.top_n_coeffs = top_n_coeffs

    @property
    def label_configuration(self) -> LabelConfiguration:
        return LabelConfiguration(labels=[Label(self.signal_name, [True, False])])

    @property
    def signal(self) -> Signal:
        return make_signal(**self.sim_config.signal)

    @property
    def batch_signal(self) -> Signal:
        return make_signal(**self.sim_config.batch_signal)

    @property
    def signal_name(self) -> str:
        return self.signal.id

    @property
    def col_mapping(self) -> dict:
        return {0: 'sequence_aas', 1: 'v_genes', 2: 'j_genes', 3: self.signal_name, 4: 'implanted', 5: 'batch'}

    @property
    def meta_col_mapping(self) -> dict:
        return {self.signal_name: self.signal_name, 'batch': 'batch', 'implanted': 'implanted'}

    def run(self, path: Path):

        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

        EnvironmentSettings.set_cache_path(path / 'cache')

        setting_paths = []

        for correction in self.sim_config.batch_corrections:
            res_path = self.run_for_impl_setup(PathBuilder.build(path / 'batch'), self.sim_config.implanting_config.batch, correction)
            setting_paths.append(res_path)

        res_path = self.run_for_impl_setup(PathBuilder.build(path / 'control'), self.sim_config.implanting_config.control, correct=None)
        setting_paths.append(res_path)

        shutil.rmtree(path / 'cache')

        self.make_reports(setting_paths, path)

    def make_reports(self, setting_paths: List[Path], result_path: Path):
        make_summary(setting_paths, result_path)

        enriched_kmer_df = self._make_enriched_kmer_summary_df(setting_paths)
        plot_enriched_kmers(result_path, enriched_kmer_df, self.sim_config.k)

    def _make_enriched_kmer_summary_df(self, paths: List[Path]) -> pd.DataFrame:

        dfs = []
        for setting_path in paths:
            files = [setting_path / f"repetition_{rep}/assessment/enriched_kmer_metrics.tsv" for rep in range(1, self.sim_config.repetitions + 1)]
            df = merge_dfs(files, "repetition", setting_path.parent.name + "_" + setting_path.name)
            dfs.append(df)

        return pd.concat(dfs, axis=0)

    def run_for_impl_setup(self, path: Path, impl_setting: ImplantingSetting, correct):
        setting_path = PathBuilder.build(path / self.get_folder_name_from_correction(correct))

        logging.info(f"Starting run for implanting_group: {impl_setting.to_dict()}, correct={self.get_folder_name_from_correction(correct)}")

        with Pool(self.num_processes) as pool:  # run self.num_processes repetitions in parallel
            all_metrics = pool.starmap(self.run_one_repetition,
                                       [(PathBuilder.build(setting_path / f'repetition_{repetition + 1}'), impl_setting, correct, repetition)
                                        for repetition in range(self.sim_config.repetitions)])

        write_to_file(pd.DataFrame(all_metrics), setting_path / 'metrics.tsv')

        logging.info(f"Finished run for implanting group: {impl_setting.to_dict()}, correct={self.get_folder_name_from_correction(correct)}")

        return setting_path

    def run_one_repetition(self, path: Path, impl_setting: ImplantingSetting, correct: bool, repetition_index: int) -> dict:

        # simulate data
        train_dataset = self.simulate_data(PathBuilder.build(path / 'train_dataset'), impl_setting.train, 'train', impl_setting.name)
        test_dataset = self.simulate_data(PathBuilder.build(path / 'test_dataset'), impl_setting.test, 'test', impl_setting.name)

        # encode data for ML and optionally correct for batch effects (if correct is not None)
        train_dataset, test_dataset = self.encode_with_kmers(train_dataset, test_dataset, path / 'encoding', correct)

        # train and asses an ML model (here always logistic regression)
        log_reg_info_path = PathBuilder.build(path / 'assessment')
        log_reg = self.train_log_reg(train_dataset, log_reg_info_path)
        metrics = self.assess_log_reg(log_reg, test_dataset, log_reg_info_path)

        return {**metrics, **{'repetition': repetition_index + 1}}

    def simulate_data(self, path: Path, impl_group: ImplantingGroup, name: str, setting_name: str) -> SequenceDataset:

        motifs = re.compile("|".join([motif.seed[0] + "[A-Z]" + motif.seed[2]
                                      if any(key != 0 and val > 0 for key, val in motif.instantiation._hamming_distance_probabilities.items())
                                      else motif.seed for motif in self.signal.motifs]))

        df = generate_sequences(olga_model_name=self.sim_config.olga_model_name, impl_group=impl_group, signal=self.signal, path=path,
                                setting_name=setting_name, skip_motifs=motifs,
                                batch_signal=self.batch_signal if setting_name != 'control' else None)

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

    def encode_with_kmers(self, train_dataset: SequenceDataset, test_dataset: SequenceDataset, path: Path, correct) -> Tuple[SequenceDataset, SequenceDataset]:

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

    def correct_encoded_for_batch(self, train_dataset: SequenceDataset, path: Path, correct) -> SequenceDataset:

        if correct is not None:
            PathBuilder.build(path)

            corrected_train = self._correct_encoded_dataset(train_dataset, correct, path)
            self._report_correction_diff(train_dataset.encoded_data, corrected_train.encoded_data, path)

            return corrected_train
        else:
            return train_dataset

    def _report_correction_diff(self, train_encoded: EncodedData, corrected_train_encoded: EncodedData, path: Path):

        baseline_df = pd.DataFrame(train_encoded.examples, columns=[f"baseline_{feature}" for feature in train_encoded.feature_names])
        correction_df = pd.DataFrame(columns=[f"corrected_{feature}" for feature in corrected_train_encoded.feature_names], data=corrected_train_encoded.examples)
        write_to_file(pd.concat([baseline_df, correction_df], axis=1), path / f"train_corrected.tsv")

    def _correct_encoded_dataset(self, dataset: SequenceDataset, correction, path: Path) -> SequenceDataset:
        corrected_dataset = dataset.clone()
        meta_train_df = dataset.get_metadata([self.signal_name, 'batch'], return_df=True)
        meta_train_df[self.signal_name] = [1 if el == 'True' else 0 for el in meta_train_df['signal']]
        meta_train_df['batch'] = [1 if el == 'True' else 0 for el in meta_train_df['batch']]
        write_to_file(meta_train_df, path / 'correction_meta_overview.tsv')

        correction_coefficients = {"feature": [], "correction": [], "signal_contribution": []}

        for feature_index, feature in enumerate(dataset.encoded_data.feature_names):

            y = dataset.encoded_data.examples[:, feature_index].flatten().tolist()[0]
            lin_reg = copy.deepcopy(correction)
            lin_reg.fit(meta_train_df.values, y)

            correction_coefficients['correction'].append(lin_reg.coef_[1])
            correction_coefficients['feature'].append(feature)
            correction_coefficients['signal_contribution'].append(lin_reg.coef_[0])

            corrected_y = copy.deepcopy(np.array(y))
            corrected_y[meta_train_df['batch'] == 1] = corrected_y[meta_train_df['batch'] == 1] - lin_reg.coef_[1]
            corrected_dataset.encoded_data.examples[:, feature_index] = corrected_y.reshape(-1, 1)

        correction_coefficients = pd.DataFrame(correction_coefficients)
        write_to_file(correction_coefficients, path / 'correction_coefficients.tsv')

        return corrected_dataset

    def train_log_reg(self, train_dataset: SequenceDataset, path: Path) -> LogisticRegression:
        log_reg = LogisticRegression(penalty="l1", solver='saga', max_iter=1000)

        clf = GridSearchCV(estimator=log_reg, n_jobs=self.num_processes, param_grid={"C": [1, 0.1, 0.01, 0.001, 0.0001]}, scoring='balanced_accuracy',
                           cv=3, verbose=0)

        clf.fit(train_dataset.encoded_data.examples, train_dataset.encoded_data.labels[self.signal_name])

        bacc = balanced_accuracy_score(train_dataset.encoded_data.labels[self.signal_name], clf.predict(train_dataset.encoded_data.examples))
        logging.info(f"Training balanced accuracy: {bacc}")

        log_reg = clf.best_estimator_

        save_to_yaml(vars(log_reg), path / 'log_reg.yaml')

        return log_reg

    def assess_log_reg(self, logistic_regression: LogisticRegression, test_dataset: SequenceDataset, path: Path):
        y_pred = logistic_regression.predict(test_dataset.encoded_data.examples)
        true_y = test_dataset.encoded_data.labels[self.signal_name]

        self._summarize_predictions(test_dataset, y_pred, path)

        tn, fp, fn, tp = confusion_matrix(true_y, y_pred).ravel()
        labels = self.label_configuration.get_label_values(self.signal_name)

        metrics = {
            'specificity': float(tn / (tn + fp)),
            'sensitivity': float(recall_score(true_y, y_pred, labels=labels)),
            'balanced_accuracy': float(balanced_accuracy_score(true_y, y_pred)),
            'auc': float(roc_auc_score(true_y, y_pred, labels=labels))
        }

        save_to_yaml(metrics, path / 'metrics.yaml')
        plot_log_reg_coefficients(logistic_regression, test_dataset.encoded_data.feature_names, self.top_n_coeffs, self.signal.motifs, path)

        sorted_indices = np.argsort(np.abs(logistic_regression.coef_.flatten()))[-self.top_n_coeffs:]
        enriched_kmers = np.array(test_dataset.encoded_data.feature_names)[sorted_indices]
        kmer_df = pd.DataFrame(self._assess_kmer_discovery(self.signal.motifs, enriched_kmers, self.sim_config.k))
        write_to_file(kmer_df, path / 'enriched_kmer_metrics.tsv')

        return metrics

    def _summarize_predictions(self, dataset: SequenceDataset, predictions, path: Path):
        df = dataset.get_metadata(['signal', 'implanted'], return_df=True)
        df['predicted'] = predictions

        write_to_file(df, path / 'test_predictions.tsv')

        count_summary_df = df.value_counts(sort=False).reset_index().rename(columns={0: 'counts'})\
            .pivot_table(values='counts', index=['signal', 'implanted'], columns=['predicted'])\
            .rename(columns={0: 'predicted_0', 1: 'predicted_1'}).reset_index()

        write_to_file(count_summary_df, path / 'count_summary.tsv')

    def get_folder_name_from_correction(self, correct) -> str:
        if correct is None:
            return "no_correction"
        elif isinstance(correct, Ridge):
            return f"ridge_{correct.alpha:.0e}"
        else:
            return "linear_reg"
