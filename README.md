# DecouplePhy: An Industrial Anomaly Detection Method Based on the Fusion of Physical Manifolds and Nonlinear Residual Features

This repository contains the complete artifact, source code, and evaluation scripts for **DecouplePhy**, a novel dual-track industrial anomaly detection framework designed to handle stealthy cyber-attacks amidst complex physical process transients. The architecture is explicitly engineered for lightweight edge-side computing nodes, achieving deterministic linear time complexity O(N) for system identification.

Our core model contributions are located at:
`code/ipal-ids-framework/ids/DecouplePhy/`

---

## 📂 Repository Architecture

```text
.
├── code/
│   ├── ipal-ids-framework/      # Extended industrial intrusion detection system core
│   │   └── ids/DecouplePhy/     # Macroscopic physical manifold & microscopic Transformer
│   └── ipal-evaluate/           # Evaluation metrics suite (including eTaPR)
├── config/                      # Configuration files with hyperparameter bindings for all datasets
├── ablation/                    # Configuration and models for Track 1 and Track 2 ablation studies
├── related-work/                # Baseline models and state files (GeCo, PASAD, Seq2SeqNN, etc.)
├── knowledge-based/             # Expert invariant configurations and outputs
├── datasets/                    # Instructions for standardized IPAL format evaluation tensor logs
├── results/                     # Pre-computed evaluation reports (.json) and figures (.pdf)
├── eval_baseline_PureTransformer.py  # Evaluates Track 2 ablation & calculates RCA HR@k scores
├── eval_diagnostics.py          # Generates high-resolution spatial-temporal residual heatmaps
├── plot_main_performance.py     # Plots overall F1 & eTaF1 detection performance comparison
└── plot_ablation_study.py       # Plots ablation study variant performance comparisons



Environment Configuration & Installation
We highly recommend utilizing a clean Python virtual environment to isolate dependencies. Execute the following sequential commands to build the exact runtime workspace:

# 1. Create and activate a virtual environment
python3 -m venv venv
source ./venv/bin/activate

# 2. Install core visualization and machine learning dependencies
pip3 install igraph==0.10.4
pip3 install pandas numpy torch matplotlib seaborn

# 3. Install the extended IPAL framework in editable mode
pip3 install code/ipal-ids-framework/
pip3 install code/ipal-evaluate/


Dataset Requirements
Due to strict licensing and confidentiality agreements, raw network capture or sensory logs for SWaT, HAI, and BATADAL cannot be distributed directly. Acquire the official text datasets from their respective testbed authorities and transcribe them into the standardized IPAL data formatting logs using the public utilities available here:
https://github.com/fkie-cad/ipal_datasets

Execution & Verification Pipeline
You can completely reproduce all baseline logs, evaluation metrics, and peer-review publication figures using the dedicated scripts listed below.

Note: The generated log files (.txt), appendix reports (.json), and figures (.pdf) will be saved directly in the root directory.

1. Model Training & Inference (Core Execution)
Before generating evaluation plots, execute the main intrusion detection script to train the models and perform inference on the test sets. You must specify the target dataset as an argument.

# Example: Run DecouplePhy and baselines on the BATADAL dataset

python3 run-ids.py BATADAL

# Available dataset arguments:
# {SWaT, HAI, BATADAL}


2. Macro Anomaly Detection Performance
This evaluation relies on compiled metrics against alternative IIDS paradigms including GeCo, SIMPLE, TABOR, Invariant, Seq2SeqNN, and PASAD on BATADAL, HAI, and SWaT.

python3 plot_main_performance.py

Output: Generates high-contrast academic bar charts validating DecouplePhy's superior trade-off in balancing traditional point-wise F1 and continuity-aware eTaF1 boundaries.


3.Multi-Track Ablation Study
Validates the structural significance of isolating macroscopic background noise before passing residuals into deep attention models.

python3 plot_ablation_study.py

Output: Quantifies and plots exact metrics for the full DecouplePhy architecture alongside isolated tracks: w/o Macro-Phy (pure self-attention) and w/o Micro-Attention (pure first-order physics bounds).


4. Microscopic Residual Maps & Spatial-Temporal Analysis
Generates diagnostic visualizations demonstrating how DecouplePhy suppresses noise spikes during process changes while pinpointing cross-sensor co-deviations under stealthy operations.

python3 eval_diagnostics.py

Output: Exports high-resolution PDF matrices (Result_<Dataset>_Academic.pdf) mapping monitored components against sliding sequence windows.

5. Root Cause Analysis (RCA) & Attribution Hits
Evaluates how targeted ablation configurations affect continuous target isolation based on Strict Entity-Level Matching and Topology-Aware Matching.

python3 eval_baseline_PureTransformer.py

Output: Records full diagnostics in RCA_Event_Log_AblationTrack2_<Dataset>.txt and dumps structural JSON tables with $HR@1$, $HR@3$, and $HR@5$ validation metrics.

