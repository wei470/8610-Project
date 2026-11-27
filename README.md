# Building Energy Prediction with Gaussian RBF-KAN and MLP

This project predicts two simulated building loads, heating and cooling, from the UCI Energy Efficiency dataset. The KAN variant represents each connection with 11 weighted Gaussian radial basis functions. The centers range from −2.5 to 2.5 in steps of 0.5. The Gaussian scale is 0.5.

## Project files

| File | Contents |
|---|---|
| `Energy_KAN.ipynb` | Gaussian RBF-KAN training, evaluation, and a worked example that reconstructs one learned connection. |
| `Energy_MLP.ipynb` | MLP training and evaluation with the same data splits. |
| `Energy_No_Input_Standardization.ipynb` | Evaluates both models with and without standardizing the six numeric inputs. |
| `ENB2012_data.xlsx` | Original UCI Energy Efficiency dataset. |
| `KAN_Preprocessed_16_Inputs.xlsx` and `MLP_Preprocessed_16_Inputs.xlsx` | The 16 model inputs for both test protocols, in the original workbook's row order. |
| `compare_results.py` | Combines both models’ outputs and draws the comparison figures. |
| `results/` | Configuration trials, per-run scores, predictions, and plots. `results/kan/` and `results/mlp/` hold the separate outputs. |
| `mlp-building-walkthrough.html` | Four-step offline MLP demonstration for Excel row 3, using a seed-101 reproduction of the random-row model at epoch 242. |
| `gaussian-edge-animation.html` | Offline demo that adds learned Gaussian terms one at a time. |
| `Presentation_中文版.pptx` | Chinese presentation with validation MSE curves, early-stopping examples, and a four-panel test prediction comparison. |
| `Presentation.pptx` | 18 slides with speaker notes, input examples on slides 7 and 8, and a link to the demo on slide 10. |

## Experimental splits

Random rows use 538 training, 115 validation, and 115 test records. The held-out shape split uses 8 training shapes (512 records), 2 validation shapes (128), and 2 test shapes (128). Both models use the same assignments.

The earlier combined notebook (`Energy_KAN_MLP.ipynb`), `Report.tex`, and the LaTeX tables under `results/` reflect an earlier experiment. The standalone notebooks and current CSV results document the current experiment.

## Reproduce the experiments

In a Python 3.12 environment, open a terminal in the project folder and run:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open `Energy_KAN.ipynb` and `Energy_MLP.ipynb` and select **Restart Kernel and Run All Cells**. Then run `python compare_results.py` to regenerate the comparison files in `results/`. Training times vary by machine and background activity.

## Presentation plan

December 8, 2026, at 8:30 a.m. The Chinese deck has 17 slides and budgets 23 minutes for the talk, including transitions. Keep both HTML demos next to the PPTX when copying it to the presentation computer. Slide 8 opens the MLP walkthrough and slide 9 opens the Gaussian KAN demo. The remaining 7 minutes are for questions.

## Sources

- Tsanas and Xifara (2012). [Energy Efficiency](https://doi.org/10.24432/C51307), UCI Machine Learning Repository. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Liu et al. [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756).
- Li (2024). [Kolmogorov-Arnold Networks are Radial Basis Function Networks](https://arxiv.org/abs/2405.06721).
