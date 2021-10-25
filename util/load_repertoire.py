from pathlib import Path

from immuneML.IO.dataset_import.OLGAImport import OLGAImport
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.util.ImportHelper import ImportHelper
from immuneML.util.PathBuilder import PathBuilder
from pandas import Series


def load_olga_repertoire(filepath: Path, result_path: Path):
    from immuneML.IO.dataset_import.DatasetImportParams import DatasetImportParams

    default_params = DefaultParamsLoader.load('datasets', 'olga')
    params = DatasetImportParams(**default_params, **{'path': filepath.parent, 'result_path': result_path})

    PathBuilder.build(result_path / 'repertoires')

    repertoire = ImportHelper.load_repertoire_as_object(OLGAImport, metadata_row=Series({'filename': filepath.name}), params=params)
    repertoire.metadata['filename'] = repertoire.data_filename.name

    return repertoire


def load_iml_repertoire(filepath: Path, identifier: str):

    if filepath.is_dir():
        data_filename = filepath / f"{identifier}.npy"
    else:
        data_filename = filepath

    repertoire = Repertoire(data_filename=data_filename, metadata_filename=data_filename.parent / f'{identifier}_metadata.yaml',
                            identifier=identifier)

    return repertoire
