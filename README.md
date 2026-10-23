# Koopman-online-updated-MPC

Code and data accompanying the paper **“Learning Model Predictive Control of Nonlinear Systems with Time-varying Parameters Using Koopman Operator.”**

## Code organization and quick start

The code has been reorganized to make the three system examples easier to run. The [`Launcher`](./Launcher) directory contains MATLAB entry scripts and their accompanying weights for:

| System | Main script |
|---|---|
| Duffing oscillator | `Duffing_TerminalWeight_Trial_Script.m` |
| Van der Pol oscillator | `VDP_TerminalWeight_Trial_Script.m` |
| Cascaded tanks | `Tank_System.m` |

To run an example, open its corresponding folder under `Launcher` in MATLAB, set that folder as the current directory, and run the main script. Simulation parameters can be adjusted at the beginning of each script.

The LMI-based Duffing and Van der Pol scripts require **YALMIP** and **MOSEK**. The cascaded tanks script uses MATLAB’s `quadprog` and `poslin` functions.

The original implementation files remain in their existing locations. The reorganized MATLAB scripts are provided as supplementary implementations; the earlier versions and their commit history remain available for reference.

## Original implementations and training

- `duffing.py` and `vanderpol.py` contain the original Python simulation and plotting workflows for the Duffing and Van der Pol examples.
- `DeepLearning_KoopmanControl_Approach3.py` is the neural-network training script.
- The `.pkl` files contain trained encoder and decoder parameters, $\theta_E$ and $\theta_D$.
- MATLAB-compatible weights are supplied as `.mat` files.
- The repository also includes a cascaded tanks example and a one-dimensional system example with lifting functions trained using KMAE.

## Erratum: encoder dimension for the cascaded tanks example

In Table 2 of the paper, the encoder architecture for the cascaded tanks example is listed as **2 → 100 → 100 → 6**.

The encoder supplied in this repository (`Encoder_Tank.m`, loading `Weights/Tank_New.mat`) actually uses **2 → 100 → 100 → 10**, producing a **10-dimensional lifted representation**.

The encoder configuration in Table 2 should therefore read **2 → 100 → 100 → 10** to match the released implementation.
