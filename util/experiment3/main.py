from datetime import datetime

from util.SimConfig import SimConfig, make_signal, ImplantingConfig, ImplantingGroup, ImplantingUnit
from util.dataset_util import setup_path
from util.experiment3.experiment3 import Experiment3
from util.util import write_config

if __name__ == "__main__":

    config = SimConfig(k=4, repetitions=10, p_noise=0.1, olga_model_name='humanTRB',
                       signal=make_signal(motif_seeds=["YEQ", "PQH", "LFF"], seq_position_weights={108: 0.5, 109: 0.5}),
                       fdrs=[0.05], sequence_encoding='continuous_kmer',
                       implanting_config=ImplantingConfig(
                           control=None,
                           batch=ImplantingGroup(baseline=ImplantingUnit(0.9, []), name='batch', seq_count=1000,
                                                 modified=ImplantingUnit(0.1, ["TRBV20", "TRBV5-1", "TRBV24", "TRBV27"]))))

    path = setup_path(f"/Users/milenpa/PycharmProjects/CausalAIRR/experiment3/AIRR_classification_{datetime.now()}")
    write_config(config, path)

    experiment = Experiment3(config, 0.01)
    experiment.run(path)
