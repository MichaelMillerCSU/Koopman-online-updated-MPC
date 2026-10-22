# Koopman-online-updated-MPC

# Code and Data for paper "Learning Model Predictive Control of Nonlinear Systems with Time-varying Parameters Using Koopman Operator"

Run duffing.py and vanderpol.py can generate the figures in the paper.

The training file is DeepLearning_KoopmanControl_Approach3.py.

pkl files are the parameters of \theta_D and \theta_E of duffing system and van der pol system for different systems.

New example Cascaded Tank System is added.

The parameters of theta_D and \theta_E are transfer to mat file in Weights folder

An One-dimensional system example is added, with lifted function trained by KMAE.

## Erratum: encoder dimension for the cascaded tanks example

In Table 2 of the paper, the encoder architecture for the cascaded tanks example is listed as 2 → 100 → 100 → 6.

The encoder supplied in this repository (Encoder_Tank.m, loading Weights/Tank_New.mat) actually uses 2 → 100 → 100 → 10, producing a 10-dimensional lifted representation.

The encoder configuration in Table 2 should therefore read 2 → 100 → 100 → 10 to match the released implementation.
