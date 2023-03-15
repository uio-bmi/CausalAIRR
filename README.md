# Improving generalization of machine learning-identified biomarkers with causal modeling: an investigation into immune receptor diagnostics 

## Experiment 1: Confounder influence on the ML performance

In this experiment, the causal graph consists of three nodes: the immune state, AIRR, the confounder and the selection node. The selection node was 
used to make the dataset balanced and depended only on the immune state. We explore three scenarios within the same causal graph: 

1. The confounder distribution remains stable between source and target population (between training and test),
2. The confounder distribution changes between source and target population, but it does not influence the performance of the ML model, and
3. The confounder distribution changes between source and target population, and it also influences the performance of the ML model.

To show the influence of a confounder and its distributional changes on the ML model, we first simulated the AIRR data from the causal graph using 
DagSim (https://github.com/uio-bmi/dagsim) [1]. The confounder and the immune state follow a binomial distribution, but the value of the immune state 
distribution parameter depends on the value of the confounder. AIRRs were simulated by first generating a naive repertoire with OLGA [2]. To simulate 
the influence of the confounder on AIRR, we implanted one of 3 possible 3-mers into a certain percentage of sequences in the AIRR. For AIRRs 
simulated in this way, the immune state signal was implanted where the immune state corresponded to “diseased”. The same causal graph was used to 
simulate both source and target distribution of all variables. 

For the ML task, we used k-mer frequency encoding of the AIRRs and logistic regression for prediction. We used the immuneML platform with fixed 
training and test set (corresponding to the data from source and target populations) to train the ML model, examine what the model has learned, and 
compare the learned signals across the source and target populations.

In the second scenario, we used the same causal graph and simulation procedure as before, except that the confounder distribution was changed between 
the source and target population. This change was implemented by using different parameter values of the binomial distribution when simulating the 
confounder. The ML approach was also the same.

In the third scenario, we used the same simulation procedure but increased the difference between the confounder distribution in the source and 
target population (represented here via training and test set). This larger difference resulted in worse performance on the test set.

We repeated the simulation and ML analysis 30 times to obtain performance bounds.

The experiment was set up and run as defined in `AIRR_experiment_1.py`. Relevant code is located under `causal_airr_scripts/experiment1`. The results were deposited in Zenodo database: link.

## Experiment 2: Selection bias and batch effects make ML models susceptible to learning spurious correlations

In the second experiment, we examined how selection bias might influence the ML models in presence of batch effects in two different scenarios 
described below. The causal graph consists of the immune state, AIRR, experimental protocol, hospital, and the selection node. The encoding was again
3-mer frequencies and logistic regression was used for prediction.

In the first scenario, we explored what happens if selection introduced correlation between the hospital (that determines the experimental protocol) 
and the immune state. The training set consisted of data collected under this selection condition, while the test data did not have selection bias.

In the second scenario, there was no immune state signal implanted in the AIRRs, but due to the presence of selection bias, the ML model learned 
the spurious correlation between the AIRR and the experimental protocol. When applied to unbiased training dataset, the prediction performance 
was random.

The experiment was set up and run as defined in `AIRR_experiment_2.py`. Relevant code is located under `causal_airr_scripts/experiment2`. The results were deposited in Zenodo database: link.

## Experiment 3: Batch effects in AIRR settings as compared to classic molecular biomarkers

In the third experiment, we illustrate batch effects in two scenarios: transcriptomics and AIR classification. 

For the AIRR scenario, we simplified  the setup compared to the first two experiments and examined immune receptor classification where we aimed 
to predict receptor specificity, i.e., if a receptor will bind to e.g., a virus of interest. We simulated 5000 receptors and implanted 3-mers in the middle of receptor sequence to simulate receptor specificity and implanted different 3-mers in the beginning of the sequence to simulate the batch effects (the implanted 3-mers were different and non-overlapping so that they cannot be confused for that purpose in the subsequent analysis). 

We assessed the performance of a predictive model (lasso-regularized logistic regression) in predicting the simulated immune signal new test data in three different scenarios: (1) where no batch effects exist in the training data (control), (2) where batch effects 
exist in the training data and not removed and (3) where batch effects exist in the training data and removed. We used k-mer frequencies and logistic regression to assess ML performance for this scenario. The batch effects, when present, were highly correlated with the immune signal. We repeated the analysis 5 times.

The code for this scenario is available in `AIRR_experiment_3.py` and the relevant code is included in `causal_airr_scripts/experiment3/`. The results were deposited in Zenodo database: link.

For the transcriptomic data, we simulated RNA-seq count datasets with batch and biological effects, where both the batch and biological effects are 
known to influence the mean and dispersion of gene expression counts. The magnitude of batch effects and true biological effects for the simulations 
were chosen based on the known levels of batch and biological effects from real-world experimental datasets as described by Zhang and colleagues [3]. When simulating datasets with batch effects, we allowed the biological condition to correlate highly 
with batches, where 80% of observations of a biological condition are processed in one batch and vice versa. This represents a common scenario in 
real-world experiments, where often a large majority of the cases are processed in one batch while a large majority of controls are processed in another batch. 

We assessed the performance of a predictive model (lasso-regularized logistic regression) in predicting the true biological condition of unseen test
observations based on gene expression data in three different scenarios: (1) where no batch effects exist in the training data, (2) where batch effects 
exist in the training data but not removed and (3) where batch effects exist in the training data and removed. We repeated the experiments using ten 
independent datasets in each scenario to provide bounds for the performance metric that we computed. We assessed the accuracy of the statistical testing 
procedure in identifying the true positives and rejecting the true negatives. In addition to the findings presented in the original manuscript, all the 
supplementary results of this part of the experiment were deposited in Zenodo database: https://zenodo.org/record/7727894. Notably, the study design 
settings such as the degree of correlation between batches and biological condition, the magnitude of batch and biological effects and so on can impact the
performance of the predictive models.


# Preprint

Pavlović, M., Hajj, G. S. A., Pensar, J., Wood, M., Sollid, L. M., Greiff, V., & Sandve, G. K. (2022). Improving generalization of machine learning-identified biomarkers with causal modeling: An investigation into immune receptor diagnostics. ArXiv:2204.09291 [Cs, q-Bio]. http://arxiv.org/abs/2204.09291

# References

[1] Al Hajj, G. S., Pensar, J., & Sandve, G. K. (2022). DagSim: Combining DAG-based model structure with unconstrained data types and relations for flexible, transparent, and modularized data simulation (arXiv:2205.11234). arXiv. https://doi.org/10.48550/arXiv.2205.11234

[2] Sethna, Z., Elhanati, Y., Callan, C. G., Walczak, A. M., & Mora, T. (2019). OLGA: Fast computation of generation probabilities of B- and T-cell receptor amino acid sequences and motifs. Bioinformatics, 35(17), 2974–2981. https://doi.org/10.1093/bioinformatics/btz035

[3] Zhang, Y., Parmigiani, G., & Johnson, W. E. (2020). ComBat-seq: Batch effect adjustment for RNA-seq count data. NAR Genomics and Bioinformatics, 2(3), lqaa078. https://doi.org/10.1093/nargab/lqaa078

