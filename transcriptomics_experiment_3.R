# install and load the library that contains code for simulation and analysis presented in this script

devtools::install_github("https://github.com/KanduriC/transcriptomicsBatchEffectSim.git")
library(transcriptomicsBatchEffectSim)

set.seed(2023)
n_times <- 10 # number of replications of experiment

# parameter config for simulations, where batch effects exist
sim_params_with_batch <-
  list(
    approx_n_genes = 5000,
    n_examples = 60,
    batch1_indices = 1:30,
    batch2_indices = 31:60,
    batch1_group1_indices = 1:24,
    batch1_group2_indices = 25:30,
    batch2_group1_indices = 31:36,
    batch2_group2_indices = 37:60,
    group1_indices = c(1:24, 31:36),
    group2_indices = c(25:30, 37:60),
    batch_difference_threshold = 1,
    n_true_diff_genes = 500,
    n_true_upreg_genes = 250,
    avg_log2FC_between_groups = 1,
    train_split_prop = 0.70,
    batch_effects_exist = TRUE
  )

# parameter config for simulations, where batch effects do not exist
sim_params_without_batch <-
  list(
    approx_n_genes = 5000,
    n_examples = 60,
    group1_indices = 1:30,
    group2_indices = 31:60,
    n_true_diff_genes = 500,
    n_true_upreg_genes = 250,
    avg_log2FC_between_groups = 1,
    train_split_prop = 0.70,
    batch_effects_exist = FALSE
  )

# repeating the simulations and analyses n_times with specified parameter configurations
res_with_batch <- rep_simulations(sim_params_list = sim_params_with_batch, 
                                  n_times = n_times)
res_without_batch <- rep_simulations(sim_params_list = sim_params_without_batch, 
                                     n_times = n_times)

# accessing results presented in the manuscript. Detailed results were deposited in the Zenodo database (https://zenodo.org/record/7727894) and can be accessed by inspecting "res_with_batch" and "res_without_batch" objects.

logistic_results <- join_results(res_with_batch, res_without_batch, "logistic_results") %>% dplyr::filter(metric!="logistic_metrics.auc")

logit_coefficients <- join_results(res_with_batch, res_without_batch, "log_coef_overlap_stats")