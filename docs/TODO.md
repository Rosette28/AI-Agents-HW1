# Atomic Task Tracker: HW1 Professional Signal Denoising

## Phase 1: Project Infrastructure & Mandatory Documentation
**Lead:** Partner A

### 1.1 Environment & Git Setup
- [x] 1.1.1 Initialize local directory with `uv init`.
- [x] 1.1.2 Set Python version to 3.12 in `.python-version`.
- [x] 1.1.3 Add `torch` for neural network implementation.
- [x] 1.1.4 Add `numpy` for signal math.
- [x] 1.1.5 Add `matplotlib` and `seaborn` for visualization.
- [x] 1.1.6 Add `pytest` and `pytest-cov` for TDD compliance.
- [x] 1.1.7 Add `ruff` for static analysis.
- [x] 1.1.8 Generate `uv.lock` for deterministic environments.
- [x] 1.1.9 Configure `.gitignore` to exclude `.venv`.
- [x] 1.1.10 Configure `.gitignore` to exclude `__pycache__`.
- [x] 1.1.11 Configure `.gitignore` to exclude `.env` and `*.key`.
- [x] 1.1.12 Create `src/sdk/`, `src/services/`, `src/shared/`.
- [x] 1.1.13 Create `tests/unit/` and `tests/integration/`.
- [x] 1.1.14 Create `docs/`, `config/`, `data/`, `assets/`.
- [x] 1.1.15 Initialize `__init__.py` in all `src/` sub-folders.

### 1.2 Standards Documentation
- [x] 1.2.1 Draft `README.md` root manual.
- [x] 1.2.2 Draft `docs/PRD.md` with KPIs and math.
- [x] 1.2.3 Draft `docs/PLAN.md` with C4 diagrams.
- [x] 1.2.4 Draft `docs/TODO.md` (This document).
- [x] 1.2.5 Define `__version__ = "1.00"` in `src/shared/version.py`.
- [x] 1.2.6 Define physical constants in `src/shared/constants.py`.

### 1.3 Algorithm PRDs
- [x] 1.3.1 Create `docs/PRD_mlp.md`.
- [x] 1.3.2 Define MLP 15-node input layer (10 samples + 4 freq + 1 noise).
- [x] 1.3.3 Define MLP 10-node output layer.
- [x] 1.3.4 Create `docs/PRD_rnn.md`.
- [x] 1.3.5 Define RNN hidden state recurrence logic ($y_t = f_w(x_t, y_{t-1})$).
- [x] 1.3.6 Create `docs/PRD_lstm.md`.
- [x] 1.3.7 Define LSTM gating mechanisms (Forget, Input, Output).

---

## Phase 2: Configuration & Data Generation
**Lead:** Partner B

### 2.1 Configuration Management
- [x] 2.1.1 Create `config/setup.json`.
- [x] 2.1.2 Define 4 target frequencies: $\{f_1, f_2, f_3, f_4\}$.
- [x] 2.1.3 Define noise parameters: $\sigma$ (amplitude) and $\sigma_2$ (phase).
- [x] 2.1.4 Implement `ConfigLoader` class in `src/shared/config.py`.
- [x] 2.1.5 Function: `load_config()` (JSON parsing).
- [x] 2.1.6 Function: `validate_numeric_types()` (Defensive programming).
- [x] 2.1.7 Function: `get_frequency_vector(index)` (1-hot helper).
- [x] 2.1.8 (Docstrings, Ruff, and 150-line rule compliance).
- [x] 2.1.9 Test: Load valid `setup.json`.
- [x] 2.1.10 Test: Error on missing config file.
- [x] 2.1.11 Test: Error on invalid noise ranges (e.g., negative $\sigma$).

### 2.2 API Gatekeeper
- [x] 2.2.1 Create `src/services/gatekeeper.py`.
- [x] 2.2.2 Implement `ApiGatekeeper` class.
- [x] 2.2.3 Function: `execute_call()` (Wrapper for data requests).
- [x] 2.2.4 Function: `_enforce_rate_limit()` (Internal check).
- [x] 2.2.5 Function: `_log_transaction()` (Monitoring).
- [x] 2.2.6 Implement FIFO overflow queue.
- [x] 2.2.7 (Docstrings and compliance).
- [x] 2.2.8 Test: Successful call within limits.
- [x] 2.2.9 Test: Request queued when limit reached.

### 2.3 Dataset Generator Service
- [x] 2.3.1 Create `src/services/dataset_generator.py`.
- [x] 2.3.2 Implement `SineWaveGenerator` class.
- [x] 2.3.3 Function: `generate_pure_signal()` ($A \sin(2\pi f \phi)$).
- [x] 2.3.4 Function: `apply_noise()` ($(A \pm \sigma)$ logic).
- [x] 2.3.5 Function: `create_context_window()` (10-sample slicing).
- [x] 2.3.6 Function: `build_dataset()` (Assembling $(C, \sigma, S_c)$ pairs).
- [x] 2.3.7 Helper: `_validate_signal_length()`.
- [x] 2.3.8 (Docstrings, 150-line rule, and Ruff).
- [x] 2.3.9 Test: Signal matches math formula results.
- [x] 2.3.10 Test: 1-hot vector $C$ has correct dimensions.
- [x] 2.3.11 Test: Noisy window vs Clean window alignment.
- [x] 2.3.12 Test: Edge case (zero amplitude).
- [x] 2.3.13 Test: Edge case (zero frequency).

---

## Phase 3: Model Architecture & SDK Implementation
**Lead:** Partner A

### 3.1 Base Neural Architecture
- [x] 3.1.1 Create `src/services/base_model.py` (DRY principle).
- [x] 3.1.2 Implement `BaseNN` mixin for shared training logic.
- [x] 3.1.3 Function: `perform_training_step()`.
- [x] 3.1.4 Function: `calculate_mse_loss()`.
- [x] 3.1.5 Function: `initialize_weights()` (Xavier/He).
- [x] 3.1.6 (Compliance and Docstrings).
- [x] 3.1.7 Test: Loss decreases after training step.

### 3.2 MLP Implementation
- [x] 3.2.1 Create `src/services/models_mlp.py`.
- [x] 3.2.2 Implement `MLPModel` class.
- [x] 3.2.3 Function: `forward()` (Flattened context window).
- [x] 3.2.4 Function: `_build_layers()` (Dynamic sizing from config).
- [x] 3.2.5 Test: Input shape [Batch, 15] matches expectations.

### 3.3 RNN & LSTM Implementation
- [x] 3.3.1 Create `src/services/models_rnn.py`.
- [x] 3.3.2 Implement `RNNModel` with recurrence.
- [x] 3.3.3 Create `src/services/models_lstm.py`.
- [x] 3.3.4 Implement `LSTMModel` with gated cells.
- [x] 3.3.5 Function: `init_hidden_state()` (RNN/LSTM).
- [x] 3.3.6 Function: `init_cell_state()` (LSTM).
- [x] 3.3.7 (Compliance, 150-line checks).
- [x] 3.3.8 Test: RNN sequence processing ($t=10$).
- [x] 3.3.9 Test: LSTM gate activation variance.

### 3.4 SDK Layer
- [x] 3.4.1 Create `src/sdk/sdk.py`.
- [x] 3.4.2 Implement `SineDenoisingSDK` facade.
- [x] 3.4.3 Method: `initialize_system()` (Config + Models).
- [x] 3.4.4 Method: `execute_denoising(noisy_input)`.
- [x] 3.4.5 Method: `get_evaluation_report()`.
- [x] 3.4.6 (Single entry point enforcement).
- [x] 3.4.7 Test: SDK routes calls correctly to models.
- [x] 3.4.8 Test: End-to-end pipeline (Data -> Train -> Eval).

---

## Phase 4: Quality, Research & Submission
**Lead:** Partner B

### 4.1 Testing & Coverage
- [ ] 4.1.1 Configure `pytest-cov` in `pyproject.toml`.
- [ ] 4.1.2 Set `fail_under = 85`.
- [ ] 4.1.3 Perform Statement, Branch, and Path coverage analysis.

### 4.2 Static Analysis & Security
- [ ] 4.2.1 Execute `uv run ruff check .`.
- [ ] 4.2.2 Fix all PEP 8 and Pyflakes violations.
- [ ] 4.2.3 Verify zero secrets in code (`.env-example` check).

### 4.3 Research Notebook & Visuals
- [ ] 4.3.1 Create `data/notebooks/results_analysis.ipynb`.
- [ ] 4.3.2 Generate Line charts (Pure vs. Noisy vs. Denoised).
- [ ] 4.3.3 Generate Bar charts (MSE comparison).
- [ ] 4.3.4 Perform Sensitivity Analysis (Noise vs. Accuracy).
- [ ] 4.3.5 Write "Prompt Engineering Log".

### 4.4 Packaging & Moodle Submission
- [ ] 4.4.1 Finalize `README.md` Lab Report content.
- [ ] 4.4.2 Export Git log to verify partner contributions.
- [ ] 4.4.3 Fill Word Template with ID/Repo link.
- [ ] 4.4.4 Name file `xxxxxxxx-ex01.pdf`.
- [ ] 4.4.5 Submit the HW.