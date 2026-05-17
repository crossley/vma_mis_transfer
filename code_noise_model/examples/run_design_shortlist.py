"""Run the end-to-end design shortlist analysis."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_model.classifier_recovery import evaluate_designs_classifier
from noise_model.diagnostics import cluster_recovery_table, pairwise_confusion_table
from noise_model.diagnostics import (
    conservative_threshold_table,
    output_pair_detectability,
    output_vs_process_table,
    process_subtype_table,
)
from noise_model.design import make_aggressive_candidate_designs, save_designs
from noise_model.models import MODEL_FAMILIES
from noise_model.plotting import (
    plot_classifier_confusion,
    plot_confusion,
    plot_design,
    plot_parameter_recovery,
)
from noise_model.recovery import confusion_matrix, evaluate_designs


OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"


def _line_reference(path: Path, needle: str) -> int:
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if needle in line:
            return line_no
    return 1


def _tex_escape(value: object) -> str:
    return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_")


def _write_latex_report(
    ranking: pd.DataFrame,
    results: pd.DataFrame,
    params: pd.DataFrame,
    classifier_ranking: pd.DataFrame,
    classifier_matrices: dict[str, pd.DataFrame],
    pairwise_confusions: pd.DataFrame,
    cluster_ranking: pd.DataFrame,
    output_process_ranking: pd.DataFrame,
    process_subtype_ranking: pd.DataFrame,
    output_pair_ranking: pd.DataFrame,
    conservative_ranking: pd.DataFrame,
) -> Path:
    models_path = ROOT / "src" / "noise_model" / "models.py"
    simulate_path = ROOT / "src" / "noise_model" / "simulate.py"
    fit_path = ROOT / "src" / "noise_model" / "fit.py"
    design_path = ROOT / "src" / "noise_model" / "design.py"
    recovery_path = ROOT / "src" / "noise_model" / "recovery.py"
    classifier_path = ROOT / "src" / "noise_model" / "classifier_recovery.py"
    features_path = ROOT / "src" / "noise_model" / "features.py"

    model_line = _line_reference(models_path, "MODEL_FAMILIES")
    sim_line = _line_reference(simulate_path, "def simulate_subject")
    fit_line = _line_reference(fit_path, "def sequence_nll")
    design_line = _line_reference(design_path, "def make_candidate_designs")
    recovery_line = _line_reference(recovery_path, "def evaluate_designs")
    classifier_line = _line_reference(classifier_path, "def evaluate_designs_classifier")
    features_line = _line_reference(features_path, "def extract_features")

    rank_rows = []
    for _, row in ranking.head(5).iterrows():
        rank_rows.append(
            f"{_tex_escape(row.design)} & {row.heldout_accuracy:.2f} & {row.aic_accuracy:.2f} "
            f"& {row.bic_accuracy:.2f} & {row.mean_accuracy:.2f} \\\\"
        )

    classifier_rank_rows = []
    for _, row in classifier_ranking.head(5).iterrows():
        classifier_rank_rows.append(
            f"{_tex_escape(row.design)} & {row.classifier_accuracy:.2f} "
            f"& {row.worst_model_recall:.2f} & {row.mean_recall:.2f} \\\\"
        )

    pairwise_rows = []
    for _, row in pairwise_confusions.head(8).iterrows():
        pairwise_rows.append(
            f"{_tex_escape(row.design)} & \\texttt{{{_tex_escape(row.model_a)}}} "
            f"& \\texttt{{{_tex_escape(row.model_b)}}} "
            f"& {row.symmetric_confusion:.2f} \\\\"
        )

    cluster_rows = []
    for _, row in cluster_ranking.head(5).iterrows():
        cluster_rows.append(f"{_tex_escape(row.design)} & {row.cluster_accuracy:.2f} \\\\")

    output_process_rows = []
    for _, row in output_process_ranking.head(5).iterrows():
        output_process_rows.append(
            f"{_tex_escape(row.design)} & {row.binary_accuracy:.2f} "
            f"& {row.output_recall:.2f} & {row.process_recall:.2f} "
            f"& {row.false_process_rate:.2f} & {row.false_output_rate:.2f} \\\\"
        )

    conservative_rows = []
    conservative_top = conservative_ranking[
        conservative_ranking.false_process_target.isin([0.05, 0.10, 0.20])
    ].groupby("false_process_target", group_keys=False).head(5)
    for _, row in conservative_top.iterrows():
        conservative_rows.append(
            f"{row.false_process_target:.2f} & {_tex_escape(row.design)} "
            f"& {row.threshold:.2f} & {row.process_recall:.2f} "
            f"& {row.output_recall:.2f} & {row.false_process_rate:.2f} \\\\"
        )

    subtype_rows = []
    for _, row in process_subtype_ranking.head(5).iterrows():
        subtype_rows.append(
            f"{_tex_escape(row.design)} & {row.process_subtype_accuracy:.2f} "
            f"& {row.process_subtype_accuracy_given_process_prediction:.2f} "
            f"& {row.process_predicted_as_output:.2f} \\\\"
        )

    pair_rows = []
    best_output_process_design = output_process_ranking.iloc[0].design
    pair_subset = output_pair_ranking[output_pair_ranking.design == best_output_process_design]
    for _, row in pair_subset.iterrows():
        pair_rows.append(
            f"\\texttt{{{_tex_escape(row.process_model)}}} & {row.binary_accuracy:.2f} "
            f"& {row.output_recall:.2f} & {row.process_recall:.2f} \\\\"
        )

    model_rows = []
    for family in MODEL_FAMILIES.values():
        noise = ", ".join(family.noise_params) if family.noise_params else "none"
        model_rows.append(
            f"\\texttt{{{_tex_escape(family.name)}}} & ${family.equation}$ "
            f"& \\texttt{{{_tex_escape(noise)}}} \\\\"
        )

    best_design = ranking.iloc[0].design
    best_classifier_design = classifier_ranking.iloc[0].design
    matrix = confusion_matrix(results[results.design == best_design], criterion="heldout")
    classifier_matrix = classifier_matrices[best_classifier_design]
    confusion_rows = []
    for true_model, row in matrix.iterrows():
        cells = " & ".join(f"{value:.2f}" for value in row)
        confusion_rows.append(f"\\texttt{{{_tex_escape(true_model)}}} & {cells} \\\\")
    classifier_confusion_rows = []
    for true_model, row in classifier_matrix.iterrows():
        cells = " & ".join(f"{value:.2f}" for value in row)
        classifier_confusion_rows.append(f"\\texttt{{{_tex_escape(true_model)}}} & {cells} \\\\")
    confusion_header = " & ".join([f"\\texttt{{{_tex_escape(name)}}}" for name in MODEL_FAMILIES])
    confusion_alignment = "l" + ("r" * len(MODEL_FAMILIES))

    param_summary = (
        params[params.design == best_design]
        .groupby("parameter")
        .agg(true_value=("true_value", "first"), recovered_mean=("recovered_value", "mean"))
        .reset_index()
    )
    param_rows = [
        f"\\texttt{{{_tex_escape(row.parameter)}}} & {row.true_value:.3f} & {row.recovered_mean:.3f} \\\\"
        for _, row in param_summary.iterrows()
    ]

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
\usepackage{{hyperref}}
\usepackage{{float}}
\usepackage{{courier}}
\title{{Implicit Adaptation Noise Models and Trial-Design Identifiability}}
\author{{Generated analysis from \texttt{{code\_noise\_model}}}}
\date{{\today}}
\begin{{document}}
\maketitle

\section*{{Goal}}
The analysis asks whether trial-to-trial reaching variability requires latent
process noise beyond output noise, and if so whether process noise is best
attached to the error, retention, or bias component of implicit adaptation.
All code and generated outputs live inside \texttt{{code\_noise\_model}}.

\section*{{State Model}}
For target direction $k$, the observed reach deviation is
\[
  y = x_k + \epsilon, \qquad \epsilon \sim \mathcal{{N}}(0,\sigma_y^2).
\]
The latent state is multidimensional, with one implicit estimate per target
direction. Error at target $k$ generalizes to all targets through a Gaussian
kernel $G_{{jk}}$,
\[
  G_{{jk}} = \exp\left(-\frac{{(\theta_j-\theta_k)^2}}{{2w^2}}\right).
\]
The implementation of the target grid and candidate designs starts in
\texttt{{src/noise\_model/design.py:{design_line}}}; simulation starts in
\texttt{{src/noise\_model/simulate.py:{sim_line}}}.

\section*{{Model Families}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lll}}
\toprule
Model & Update equation & Noise parameter(s) \\
\midrule
{chr(10).join(model_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
The model metadata are implemented in
\texttt{{src/noise\_model/models.py:{model_line}}}. The simulator maps these
equations into executable updates in
\texttt{{src/noise\_model/simulate.py:{sim_line}}}.

\section*{{Fitting}}
Models are fit with a Gaussian filtering approximation. Observation likelihoods
are accumulated over reach outputs, while transition variance depends on the
candidate process-noise family. The likelihood implementation begins at
\texttt{{src/noise\_model/fit.py:{fit_line}}}. Model recovery and design
ranking are implemented from
\texttt{{src/noise\_model/recovery.py:{recovery_line}}}. The default report
uses representative parameter values for fast design triage; bounded numerical
optimization is implemented in the same fitting module for slower confirmatory
runs.

\section*{{Model-Free Recovery}}
Because exact likelihoods for multiplicative process noise are nontrivial, the
package also includes simulation-based model-free recovery. Each synthetic
dataset is converted to summary features targeting output variance, carryover,
lagged dependence, error-magnitude effects, feedback-type effects, phase
effects, and target-specific variability. Feature extraction begins at
\texttt{{src/noise\_model/features.py:{features_line}}}. A standardized
multinomial logistic classifier is trained and tested on held-out synthetic
datasets; the design-level workflow starts at
\texttt{{src/noise\_model/classifier\_recovery.py:{classifier_line}}}.

\section*{{Candidate Design Ranking}}
An expanded library of 200-trial designs was compared. Each target direction
$-30,-15,0,15,30^\circ$ appears uniformly. Synthetic data were generated from
each true model family and refit with every candidate family. Designs are
ranked by the mean of held-out, AIC, and BIC recovery accuracy. The table shows
the top five likelihood-triage designs.

\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrrr}}
\toprule
Design & Held-out & AIC & BIC & Mean \\
\midrule
{chr(10).join(rank_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section*{{Model-Free Design Ranking}}
The primary scientific question is whether variability requires latent process
noise beyond output noise. Therefore the main design ranking collapses all
process-noise families into one class and compares \texttt{{output\_only}}
against \texttt{{process}}.

\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrrrr}}
\toprule
Design & Accuracy & Output recall & Process recall & False process & False output \\
\midrule
{chr(10).join(output_process_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section*{{Conservative Process Detection}}
Because false positives are costly, designs are also ranked by process recall
under fixed false-process-rate limits. A dataset is classified as process only
when the classifier's process probability exceeds a threshold chosen on held-out
synthetic data.
\begin{{table}}[H]
\centering
\begin{{tabular}}{{rlrrrr}}
\toprule
False-process target & Design & Threshold & Process recall & Output recall & False process \\
\midrule
{chr(10).join(conservative_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section*{{Secondary Six-Way Ranking}}
The full six-way classifier asks whether the process-noise subtype is also
recoverable. This is secondary, because subtype identification is desirable but
not the main experimental target.

\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrr}}
\toprule
Design & Accuracy & Worst recall & Mean recall \\
\midrule
{chr(10).join(classifier_rank_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section*{{Best Model-Free Confusion Matrix}}
Best model-free design: \texttt{{{_tex_escape(best_classifier_design)}}}.
\begin{{table}}[H]
\centering
\scriptsize
\begin{{tabular}}{{{confusion_alignment}}}
\toprule
True & {confusion_header} \\
\midrule
{chr(10).join(classifier_confusion_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.85\linewidth]{{../outputs/{best_classifier_design}_classifier_confusion.png}}
\caption{{Model-free recovery for the top-ranked design.}}
\end{{figure}}

\section*{{Output-Versus-Process Pair Checks}}
For the top binary design, pairwise detectability against each process family is:
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrr}}
\toprule
Process family & Accuracy & Output recall & Process recall \\
\midrule
{chr(10).join(pair_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section*{{Process-Subtype Recovery}}
Among true process-noise datasets, subtype recovery remains a harder secondary
problem:
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrr}}
\toprule
Design & Subtype accuracy & Given process prediction & Process as output \\
\midrule
{chr(10).join(subtype_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section*{{Confusability Diagnostics}}
The model-free confusion matrices show a structured failure mode. The largest
pairwise confusions are:
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lllr}}
\toprule
Design & Model A & Model B & Symmetric confusion \\
\midrule
{chr(10).join(pairwise_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

If the families are collapsed into two coarse classes,
\texttt{{output\_only/error\_term/retention\_term}} versus
\texttt{{additive/bias\_term/full\_component}}, recovery improves:
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lr}}
\toprule
Design & Cluster accuracy \\
\midrule
{chr(10).join(cluster_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
This suggests that the current designs contain strong information about broad
noise structure, but not enough to reliably identify all six fine-grained
families.

\section*{{Best Design Confusion Matrix}}
Best design: \texttt{{{_tex_escape(best_design)}}}. Rows are true models; columns are
models recovered by the fast held-out likelihood triage.
\begin{{table}}[H]
\centering
\scriptsize
\begin{{tabular}}{{{confusion_alignment}}}
\toprule
True & {confusion_header} \\
\midrule
{chr(10).join(confusion_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.85\linewidth]{{../outputs/{best_design}_confusion.png}}
\caption{{Held-out model recovery for the top-ranked design.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.85\linewidth]{{../outputs/{best_design}_design.png}}
\caption{{Trial sequence for the top-ranked design.}}
\end{{figure}}

\section*{{Parameter Recovery}}
For fits where the fitted family matches the true generating family, recovered
parameters for the top design averaged:
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrr}}
\toprule
Parameter & True & Recovered mean \\
\midrule
{chr(10).join(param_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.85\linewidth]{{../outputs/{best_design}_params.png}}
\caption{{Parameter recovery for true-family fits in the top-ranked design.}}
\end{{figure}}

\section*{{Interpretation}}
The model-free classifier ranking is stronger evidence about behavioral
separability than the fixed-parameter likelihood triage. It still does not
replace full fitted likelihood recovery, but it helps identify trial structures
worth carrying forward into slower confirmatory fitting.
\end{{document}}
"""
    report_path = REPORTS / "noise_model_report.tex"
    report_path.write_text(tex)
    return report_path


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    designs = make_aggressive_candidate_designs(n_trials=200, seed=1)
    save_designs(designs, str(OUTPUTS))
    features, classifier_predictions, classifier_ranking, classifier_matrices = evaluate_designs_classifier(
        designs,
        n_reps=80,
        n_splits=20,
        seed=100,
    )
    features.to_csv(OUTPUTS / "classifier_features.csv", index=False)
    classifier_predictions.to_csv(OUTPUTS / "classifier_predictions.csv", index=False)
    classifier_ranking.to_csv(OUTPUTS / "classifier_design_ranking.csv", index=False)
    for design_name, matrix in classifier_matrices.items():
        matrix.to_csv(OUTPUTS / f"{design_name}_classifier_confusion.csv")
    pairwise_confusions = pairwise_confusion_table(classifier_matrices)
    cluster_ranking = cluster_recovery_table(classifier_predictions)
    output_process_ranking = output_vs_process_table(classifier_predictions)
    process_subtype_ranking = process_subtype_table(classifier_predictions)
    output_pair_ranking = output_pair_detectability(classifier_predictions)
    conservative_ranking = conservative_threshold_table(classifier_predictions)
    pairwise_confusions.to_csv(OUTPUTS / "pairwise_confusions.csv", index=False)
    cluster_ranking.to_csv(OUTPUTS / "cluster_recovery.csv", index=False)
    output_process_ranking.to_csv(OUTPUTS / "output_vs_process_recovery.csv", index=False)
    process_subtype_ranking.to_csv(OUTPUTS / "process_subtype_recovery.csv", index=False)
    output_pair_ranking.to_csv(OUTPUTS / "output_pair_detectability.csv", index=False)
    conservative_ranking.to_csv(OUTPUTS / "conservative_threshold_recovery.csv", index=False)

    results, params, ranking = evaluate_designs(designs, n_reps=8, seed=10, maxiter=0, fast=True)
    results.to_csv(OUTPUTS / "model_recovery_results.csv", index=False)
    params.to_csv(OUTPUTS / "parameter_recovery.csv", index=False)
    ranking.to_csv(OUTPUTS / "design_ranking.csv", index=False)

    classifier_shortlist = set(classifier_ranking["design"].head(5))
    likelihood_shortlist = set(ranking["design"].head(5))
    shortlist = classifier_shortlist | likelihood_shortlist
    for design in designs:
        if design.name in shortlist:
            plot_design(design, str(OUTPUTS / f"{design.name}_design.png"))
            plot_confusion(results, design.name, str(OUTPUTS / f"{design.name}_confusion.png"))
            plot_classifier_confusion(
                classifier_matrices[design.name],
                design.name,
                str(OUTPUTS / f"{design.name}_classifier_confusion.png"),
            )
            plot_parameter_recovery(params, design.name, str(OUTPUTS / f"{design.name}_params.png"))

    tex_path = _write_latex_report(
        ranking,
        results,
        params,
        classifier_ranking,
        classifier_matrices,
        pairwise_confusions,
        cluster_ranking,
        output_process_ranking,
        process_subtype_ranking,
        output_pair_ranking,
        conservative_ranking,
    )
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=REPORTS,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=REPORTS,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        pass

    print("Design ranking:")
    print(ranking.to_string(index=False))
    print("\nModel-free classifier design ranking:")
    print(classifier_ranking.to_string(index=False))
    print("\nCollapsed cluster recovery:")
    print(cluster_ranking.to_string(index=False))
    print("\nOutput vs process recovery:")
    print(output_process_ranking.to_string(index=False))
    print("\nProcess subtype recovery:")
    print(process_subtype_ranking.to_string(index=False))
    print("\nConservative threshold recovery:")
    print(conservative_ranking.groupby("false_process_target", group_keys=False).head(5).to_string(index=False))
    print(f"Outputs: {OUTPUTS}")
    print(f"Report source: {tex_path}")
    pdf = REPORTS / "noise_model_report.pdf"
    if pdf.exists():
        print(f"Report PDF: {pdf}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH", str(ROOT / "src"))
    main()
