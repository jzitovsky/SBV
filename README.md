# SBV

This is the code repository associated with Zitovsky et. al. 2022. If any part of this repository is used or referenced, please cite the associated paper. The purpose of these scripts are twofold: To make the results of Zitovsky et. al. 2022 reproducable, and to enable future researchers to conduct OMS experiments on the same domains as those used in Zitovsky et. al. 2022 more easily.

## Toy Experiments

The scripts needed to reproduce our toy environment results can be found in the `toy_scripts` directory of this repository. The `runExperiment.R` script takes a single command-line argument representing the generative model parameter `x` (see Appendix section C of Zitovsky et. al. 2022 for more details). For example, running the command `Rscript runExperiment.R 0.5` runs the toy experiment discussed in Zitovsky et. al. 2022 while setting x=0.5 for the generative model, including estimating the Q-functions, running SBV, running EMSBE and estimating the true MSBE,  and will create two files storing relevant information about the experiments. After running the experiment over $x\in\{0.5,0.55,0.6,0.65,0.7,0.75\}$, we can use the `plotMSBE.R` and `plotNoise.R` scripts to reproduce Figures 1 and 2 of Zitovsky et. al. 2022. 
