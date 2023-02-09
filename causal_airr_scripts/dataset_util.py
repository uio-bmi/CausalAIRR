import datetime
import os
import shutil
from glob import glob
from pathlib import Path
from typing import List
import pandas as pd
from sklearn.metrics import matthews_corrcoef

import numpy as np
from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.IO.dataset_import.OLGAImport import OLGAImport
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.util.ImportHelper import ImportHelper
from immuneML.util.PathBuilder import PathBuilder
from pandas import Series, DataFrame, read_csv


def load_olga_repertoire(filepath: Path, result_path: Path, additional_metadata: dict = None):
    from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams

    default_params = DefaultParamsLoader.load('datasets', 'olga')
    params = DatasetImportParams.build_object(**default_params, **{'path': filepath.parent, 'result_path': result_path})

    PathBuilder.build(result_path / 'repertoires')

    additional_metadata = additional_metadata if additional_metadata is not None else {}
    metadata_row = Series({**{'filename': filepath.name}, **additional_metadata})

    repertoire = ImportHelper.load_repertoire_as_object(OLGAImport, metadata_row=metadata_row, params=params)
    repertoire.metadata['filename'] = repertoire.data_filename.name

    return repertoire


def load_iml_repertoire(filepath: Path, identifier: str = None):
    assert filepath.is_file(), f"Cannot load repertoire, {filepath} is not a file."

    if identifier is None:
        identifier = filepath.stem

    repertoire = Repertoire(data_filename=filepath, metadata_filename=filepath.parent / f'{identifier}_metadata.yaml', identifier=identifier)

    return repertoire


def make_olga_repertoire(sequence_count: int, path: Path) -> Repertoire:
    olga_path = PathBuilder.build(path / 'olga')
    log_path = olga_path / "log.txt"

    seed = len(glob(str(olga_path / "*.tsv"))) + 1

    make_default_olga_repertoire(olga_path, sequence_count, seed, log_path)

    repertoire = load_olga_repertoire(filepath=Path(f"{olga_path}/{seed}.tsv"), result_path=path / "immuneML_naive")

    return repertoire


def make_default_olga_repertoire(path: Path, sequence_count: int, seed: int, log_path: Path):
    os.system(f"olga-generate_sequences --humanTRB -n {sequence_count} --seed={seed} -o {path}/{seed}.tsv >> {log_path}")


def make_dataset(repertoire_paths: List[Path], path: Path, dataset_name: str, signal_names: List[str]):
    repertoires = [load_iml_repertoire(filepath=filepath) for filepath in repertoire_paths]

    assert len(repertoires) > 0, "No repertoires in the list, cannot make dataset."

    PathBuilder.build(path)

    metadata_keys = [key for key in repertoires[0].metadata.keys() if key != 'field_list']
    metadata_file = path / f"{dataset_name}_metadata.csv"

    DataFrame({**{"subject_id": [repertoire.identifier for repertoire in repertoires]},
               **{key: [repertoire.metadata[key] for repertoire in repertoires] for key in metadata_keys}}) \
        .to_csv(path_or_buf=metadata_file, index=False)

    dataset = RepertoireDataset(labels={signal_name: [True, False] for signal_name in signal_names}, repertoires=repertoires,
                                metadata_file=metadata_file, name=dataset_name)

    return dataset


def make_AIRR_dataset(train_dataset: RepertoireDataset, test_dataset: RepertoireDataset, path: Path) -> RepertoireDataset:
    tmp_path = PathBuilder.build(path / "tmp")

    train_metadata = read_csv(train_dataset.metadata_file)
    test_metadata = read_csv(test_dataset.metadata_file)

    assert np.array_equal(train_metadata.columns, test_metadata.columns), f"Train and test metadata columns don't match."

    metadata_file = tmp_path / 'metadata.csv'
    pd.concat([train_metadata, test_metadata], axis=0).to_csv(metadata_file, index=False)

    dataset = RepertoireDataset(labels=train_dataset.labels, repertoires=train_dataset.repertoires + test_dataset.repertoires,
                                name='dataset', metadata_file=metadata_file)

    AIRRExporter.export(dataset, path, 1)

    shutil.rmtree(tmp_path)

    return dataset


def setup_path(path, remove_if_exists: bool = True) -> Path:
    path_obj = Path(path)

    if path_obj.is_dir() and remove_if_exists:
        print(f"Removing {path_obj}...")
        shutil.rmtree(path_obj)

    PathBuilder.build(path_obj)

    return path_obj


def print_all_simulation_stats(start_path: Path, columns: list, compute_correlation: bool = True):
    metadata_files = glob(str(start_path / "**/*metadata.csv"), recursive=True)

    for metadata_file in metadata_files:
        print(f"\n\n{Path(metadata_file).name}\n")
        for k, v in get_simulation_stats(metadata_file, columns, compute_correlation).items():
            print(k)
            print(f"{v}\n")


def get_simulation_stats(metadata_path: Path, columns: list, compute_correlation):
    df = pd.read_csv(metadata_path)
    output = {}
    for key in columns:
        values, counts = np.unique(df[key], return_counts=True)
        output[f"{key} stats"] = {val: count for val, count in zip(values, counts)}

    if compute_correlation and len(columns) == 2:
        try:
            output[f"Matthews correlation coefficient between {columns}"] = matthews_corrcoef(df[columns[0]], df[columns[1]])
        except Exception as e:
            print("Exception occurred while computing Matthews correlation coefficient.")

    return output


def write_to_file(df, path):
    if path.is_file():
        df.to_csv(path, sep='\t', index=None, mode='a', header=False)
    else:
        df.to_csv(path, sep="\t", index=None, header=True)


def get_dataset_from_dataframe(sequences: pd.DataFrame, file_path: Path, col_mapping: dict = None, meta_col_mapping: dict = None):
    col_mapping = {0: 'sequence_aas', 1: 'v_genes', 2: 'j_genes', 3: 'status'} if col_mapping is None else col_mapping
    meta_col_mapping = {'status': 'status'} if meta_col_mapping is None else meta_col_mapping
    sequences.to_csv(file_path, sep='\t', index=False)
    return OLGAImport.import_dataset({
        'path': file_path, 'is_repertoire': False, 'import_illegal_characters': False, 'import_empty_nt_sequences': True,
        'import_empty_aa_sequences': False, 'separator': '\t', 'region_type': 'IMGT_JUNCTION', 'columns_to_load': list(col_mapping.keys()),
        'column_mapping': col_mapping, "metadata_column_mapping": meta_col_mapping,
        'result_path': setup_path(file_path.parent / 'imported_dataset')
    }, f'seq_dataset_{datetime.datetime.now()}')
