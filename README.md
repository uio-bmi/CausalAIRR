# Improving generalization of machine learning-identified biomarkers with causal modeling: an investigation into immune receptor diagnostics 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/uio-bmi/CausalAIRR/HEAD)

## Experiment 1: Confounder influence on the ML performance

In this experiment, the causal graph consists of three nodes: the immune state, AIRR, the confounder and the selection node. The selection node was 
used to make the dataset balanced and depended only on the immune state. We explore three scenarios within the same causal graph: 

1. The confounder distribution remains stable between source and target population (between training and test),
2. The confounder distribution changes between source and target population, but it does not influence the performance of the ML model, and
3. The confounder distribution changes between source and target population, and it also influences the performance of the ML model.

To show the influence of a confounder and its distributional changes on the ML model, we first simulated the AIRR data from the causal graph using 
DagSim (https://github.com/uio-bmi/dagsim). The confounder and the immune state follow a binomial distribution, but the value of the immune state 
distribution parameter depends on the value of the confounder. AIRRs were simulated by first generating a naive repertoire with OLGA89. To simulate 
the influence of the confounder on AIRR, we implanted one of 2 possible 3-mers into a certain percentage of sequences in the AIRR. For AIRRs 
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

## Experiment 2: Selection bias and batch effects make ML models susceptible to learning spurious correlations

In the second experiment, we examined how selection bias might influence the ML models in presence of batch effects in two different scenarios 
described below. The causal graph consists of the immune state, AIRR, experimental protocol, hospital, and the selection node. The encoding was again
3-mer frequencies and logistic regression was used for prediction.

In the first scenario, we explored what happens if selection introduced correlation between the hospital (that determines the experimental protocol) 
and the immune state. The training set consisted of data collected under this selection condition, while the test data did not have selection bias.

In the second scenario, there was no immune state signal implanted in the AIRRs, but due to the presence of selection bias, the ML model learned 
the spurious correlation between the AIRR and the experimental protocol. When applied to unbiased training dataset, the prediction performance 
dropped.

Each experiment has a corresponding Jupyter notebook. Used packages are listed in the requirements.txt
file. All other code is located under util/ folder.
