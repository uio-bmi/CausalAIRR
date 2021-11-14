import os
import shutil
from pathlib import Path
from typing import List

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


def make_olga_repertoire(confounder: str, confounder_name: str, sequence_count: int, path: Path, seed) -> Repertoire:
    olga_path = PathBuilder.build(path / 'olga')
    log_path = olga_path / "log.txt"

    if confounder == "C1":
        make_default_olga_repertoire(olga_path, sequence_count, seed, log_path)
    else:
        make_modified_olga_repertoire(olga_path, sequence_count, seed, log_path)

    repertoire = load_olga_repertoire(filepath=Path(f"{olga_path}/{seed}.tsv"), result_path=path / "immuneML_naive",
                                      additional_metadata={confounder_name: confounder})

    return repertoire


def make_modified_olga_repertoire(path: Path, sequence_count: int, seed: int, log_path: Path):
    if isinstance(seed, int):
        os.system(
            f"olga-generate_sequences -n {sequence_count} --seed={seed} -o {path}/{seed}.tsv --set_custom_model_VDJ ./olga_model_removed_TRBV5_1/ >> {log_path}")
    else:
        os.system(
            f"olga-generate_sequences -n {sequence_count} -o {path}/{seed}.tsv --set_custom_model_VDJ ./olga_model_removed_TRBV5_1/ >> {log_path}")


def make_default_olga_repertoire(path: Path, sequence_count: int, seed, log_path: Path):
    if isinstance(seed, int):
        os.system(f"olga-generate_sequences --humanTRB -n {sequence_count} --seed={seed} -o {path}/{seed}.tsv >> {log_path}")
    else:
        os.system(f"olga-generate_sequences --humanTRB -n {sequence_count} -o {path}/{seed}.tsv >> {log_path}")


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
    train_metadata.append(test_metadata).to_csv(metadata_file, index=False)

    dataset = RepertoireDataset(labels=train_dataset.labels, repertoires=train_dataset.repertoires + test_dataset.repertoires,
                                name='experiment1_dataset', metadata_file=metadata_file)

    AIRRExporter.export(dataset, path, RegionType.IMGT_CDR3)

    shutil.rmtree(tmp_path)

    return dataset


def setup_path(path) -> Path:
    path_obj = Path(path)

    if path_obj.is_dir():
        print(f"Removing {path_obj}...")
        shutil.rmtree(path_obj)

    PathBuilder.build(path_obj)

    return path_obj
