# Building Energy Prediction with KAN and MLP

## Project files

| File | Contents |
|---|---|
| Report.tex | LaTeX source for the 10-page, two-column, single-spaced report. It uses figures and tables from the results folder. |
| Presentation.pptx | 15 slides with speaker notes and suggested timings for all three presenters. |
| Energy_KAN.ipynb | Standalone KAN notebook with data preparation, training, evaluation, and figures. |
| Energy_MLP.ipynb | Standalone MLP notebook with the same data preparation, evaluation, and figure layout. |
| KAN_Preprocessed_16_Inputs.xlsx and MLP_Preprocessed_16_Inputs.xlsx | Model inputs for both splits. Each sheet contains 768 rows and 16 features. |
| Energy_KAN_MLP.ipynb | Original joint benchmark notebook used for the report and presentation. |
| ENB2012_data.xlsx | Original UCI Energy Efficiency dataset. |
| requirements.txt | Python dependencies for the notebook. |
| results/ | Split assignments, configuration trials, per-run metrics, per-sample predictions, figures, and runtime details. |

## Reproduce the experiments

Open a terminal in the project folder. Use Python 3.12 and install the dependencies in a virtual environment.

    python -m pip install -r requirements.txt
    python -m jupyterlab

Open `Energy_KAN.ipynb` or `Energy_MLP.ipynb` and choose **Restart Kernel and Run All Cells**. Both notebooks use the same source data and splits, export their 16 input columns to Excel, and save results to `results/kan/` or `results/mlp/`. The report and presentation use the original joint benchmark output in `results/`. Training times can vary with hardware, software, and background activity.

## Compile the report

Run `tectonic Report.tex` from the project folder, or compile the source in a LaTeX environment with `graphicx`, `booktabs`, `caption`, and `hyperref`. Keep the PNG figures and TeX tables in `results/`. The report tables use the recorded results included with the project.

## Presentation plan

The presentation is on December 8, 2026, at 8:30 a.m. Presenter A covers slides 1–5 in 7 minutes, Presenter B covers slides 6–10 in 8 minutes, and Presenter C covers slides 11–15 in 8 minutes. The remaining 7 minutes are for questions. Open the PowerPoint speaker notes for speaking points and slide-by-slide timings.

## Data source

Tsanas, A. & Xifara, A. (2012). [Energy Efficiency](https://doi.org/10.24432/C51307). UCI Machine Learning Repository. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
