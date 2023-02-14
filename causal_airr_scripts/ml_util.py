from pathlib import Path


def define_specs(data_path: Path, experiment_name: str) -> dict:
    return {
        "definitions": {
            "datasets": {
                "dataset1": {
                    "format": 'AIRR',
                    "params": {
                        "path": str(data_path / 'full_dataset'),
                        "metadata_file": str(data_path / 'full_dataset/metadata.csv'),
                        "region_type": "IMGT_JUNCTION"
                    }
                }
            },
            "encodings": {
                "kmer_frequency": {
                    "KmerFrequency": {"k": 3}
                }
            },
            "ml_methods": {
                "logistic_regression": {
                    "LogisticRegression": {
                        "penalty": "l1",
                        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        "show_warnings": False
                    },
                    "model_selection_cv": True,
                    "model_selection_n_folds": 5
                }
            },
            "reports": {
                "coefficients": {
                    "Coefficients": {  # show top 30 logistic regression coefficients and what k-mers they correspond to
                        "coefs_to_plot": ['n_largest'],
                        "n_largest": [30]
                    }
                },
                "feature_comparison": {
                    "FeatureComparison": {
                        "comparison_label": "immune_state",
                        "show_error_bar": False,
                        "keep_fraction": 0.1,
                        "log_scale": True
                    }
                }
            }
        },
        "instructions": {
            'train_ml': {
                "type": "TrainMLModel",
                "assessment": {  # ensure here that train and test dataset are fixed, as per simulation
                    "split_strategy": "manual",
                    "split_count": 1,
                    "manual_config": {
                        "train_metadata_path": str(data_path / f"experiment{experiment_name}_train_metadata.csv"),
                        "test_metadata_path": str(data_path / f"experiment{experiment_name}_test_metadata.csv")
                    },
                    "reports": {
                        "models": ["coefficients"],
                        "encoding": ["feature_comparison"]
                    }
                },
                "selection": {
                    "split_strategy": "k_fold",
                    "split_count": 5
                },
                "settings": [
                    {"encoding": "kmer_frequency", "ml_method": "logistic_regression"}
                ],
                "dataset": "dataset1",
                "refit_optimal_model": False,
                "labels": ["immune_state"],
                "optimization_metric": "balanced_accuracy",
                "metrics": ['log_loss', 'auc']
            }
        }
    }

