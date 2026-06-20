# TattleTots Cross-Domain Simulation & Experimental Coordination

This workspace coordinates the TattleTots simulation engine with its three domain-specific adapters:
1. **Coral Key (ReefWatch)**: Fishery monitoring, IUU detection, and ocean sensor ecology.
2. **Scrapiron and the Bear (FireEcology)**: Grid-based wildfire detection and autonomous drone suppression.
3. **Xylella_SPQR (GrainGuard)**: Precision pest and weed management.

This root directory contains two primary coordination systems designed to run batch simulations, manage configuration overrides, capture execution logs, and index results.

---

## 1. Batch Execution System

The batch execution system allows running arbitrary, custom-defined configurations for any of the three domain simulations with TattleTots integration.

### Files
* **`run_batch.py`**: A robust Python script that executes simulations in parallel or sequentially, merges custom overrides with default configurations, captures console output into log files, and generates a unified index.
* **`batch_config.json`**: A JSON configuration file defining the output directory, the list of runs, and custom parameter overrides for both the TattleTots engine (`simulation`) and the domain-specific environments (`domain`).

### Commands
To run the batch simulations in parallel (recommended):
```bash
cd TattleTots/Large\ Experiments
python run_batch.py --config batch_config.json --parallel --workers 8
```

To run sequentially:
```bash
python run_batch.py --config batch_config.json
```

To override the output directory:
```bash
python run_batch.py --config batch_config.json --output-dir custom_results_dir
```

### Outputs
All outputs are saved to the configured directory (e.g., `batch_results/`):
* **`key.json`**: A summary index file mapping each run to its status, execution time, configuration file, output results file, log file, and key extracted metrics (e.g., final population, precision, total cost, and domain-specific metrics).
* **`<run_name>_config.json`**: The fully resolved configuration used for the run.
* **`<run_name>_results.json`**: The complete TattleTots simulation output JSON.
* **`<run_name>.log`**: Full standard output and standard error captured during execution.

---

## 2. Designed Experiments System (DOE)

The designed experiments system implements the full combinatorial sweeps described in `Designed Experiments.txt` and `Design of Experiment.md`. It sweeps across 5 levels of TattleTots shared parameters (Conservative to Exploratory) and domain-specific factors.

### Files
* **`run_experiments.py`**: A parallelized script that generates the full combinatorial space of runs, executes them using ProcessPoolExecutor, captures execution logs, and indexes the results.
* **`baseline_parallel.py`**: Shared ProcessPoolExecutor helpers used by batch, DOE, and domain baseline runners.
* **`designed_experiments_config.json`**: The generated configuration file defining the parameter sweeps, step counts (800 steps/epochs), triplicate seeds (42, 43, 44), and factor levels.

### Swept Factors & Mappings
* **TattleTots Shared Parameters**: Swept across 5 interpolated levels from Conservative to Exploratory (initial/max population, initial info/attention energy, false alarm penalty, trust delta negative/positive, mutation rate, recombination probability).
* **`max_stream_dim`**: Hardcoded to `1000` across all runs.
* **Coral Key Factors**: Sweeps IUU vessel counts (1, 3, 6), AIS disable probabilities, spoof probabilities, catch underreport fractions, SAR satellite revisit intervals (4, 8, 16 epochs), and platform interference rates.
* **Fire Ecology Factors**: Sweeps deployment phases (Phases 0, 1, 2, 3), sensor dropout rates (0%, 20%, 50%), and stream dimensions (400, 1000, 5000).
* **Grain Guard Factors**: Sweeps landscapes (monoculture, orchard, intercrop), sensor budgets (sparse, medium, dense), and stream dimensions (117, 500, 1000).

### Commands
To run a fast parallelized smoke test (verifies the entire pipeline in ~3 seconds):
```bash
python run_experiments.py --smoke-test --parallel
```

To run the full combinatorial suite of designed experiments:
```bash
python run_experiments.py --parallel
```

To limit execution to a single domain (e.g., Coral Key):
```bash
python run_experiments.py --domain coral_key --parallel
```

### Outputs
Results are saved to `designed_experiments_results/` (or `smoke_test_results/` for smoke tests):
* **`key.json`**: The summary index file cataloging every combinatorial run, its status, execution time, and standard/domain-specific metrics.
* **`<run_name>_config.json`**: The fully resolved configuration for that specific combinatorial run.
* **`<run_name>_results.json`**: The complete simulation output JSON.
* **`<run_name>.log`**: Full console log of the run.

---

## Workspace Directory Structure

```
d:\TotsFiles\
├── Coral_Key_in_Three_Hour_Epochs/  # ReefWatch domain repository
├── Scrapiron_and_the_Bear/          # FireEcology domain repository
├── TattleTots/                      # Shared BMA engine repository
│   └── Large Experiments/           # DOE, batch, baseline_parallel.py
├── Xylella_SPQR/                    # GrainGuard domain repository
├── batch_results/                   # Output directory for batch runs (generated)
└── smoke_test_results/              # Output directory for DOE smoke tests (generated)
```

## Baseline Parameter Scans

Non-Tots baseline scans live in each domain repo under `baselines/`. Orchestrator:

```bash
cd TattleTots/Large\ Experiments
python run_all_baselines.py --smoke-test
python run_all_baselines.py --domain coral_key --workers 8
```

See the workspace root `README.md` for install steps and run counts.

## Baseline Archive Validation

Before trusting archived A0–A3 results for BMA comparisons, spot-check equivalence against current code:

```bash
cd TattleTots/Large\ Experiments
python baseline_spot_check.py --domain all --sample 100 --workers 8
```

Output: `baseline_spot_check_results/spot_check_report.json` (same-seed determinism gate at 1% relative tolerance).

## BMA vs Baseline Comparison

After `run_quick_validation.py`, compare BMA runs to archived baselines and configured targets:

```bash
python compare_bma_baselines.py
python compare_bma_baselines.py --bma-key quick_validation_results/key.json
```

Output: `quick_validation_results/bma_baseline_comparison.json` (default paths).

## Fire COP Diagnostics

```bash
python debug_fire_dispatch.py
```

Logs COP threat levels and dispatch target counts for a fire quick-validation config (smoke/debug only).

