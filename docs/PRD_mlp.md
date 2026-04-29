# Algorithm PRD: Multilayer Perceptron (MLP)

## 1. Theoretical Background
The Multilayer Perceptron (MLP) is a foundational feedforward artificial neural network. It consists of an input layer, hidden layers, and an output layer, where every node is fully connected to all nodes in the subsequent layer. In the context of time-series denoising, the MLP processes the entire 10-sample window simultaneously. It lacks an internal memory mechanism, meaning it evaluates the temporal data as a static, flat feature set rather than a sequential flow.

## 2. Architecture & Technical Specification
- **Input Dimension:** 15 features.
- **Output Dimension:** 10 features (continuous signal prediction).
- **Proposed Architecture:**
  - Input Layer: `Linear(15, 64)`
  - Activation: `ReLU`
  - Hidden Layer: `Linear(64, 64)`
  - Activation: `ReLU`
  - Output Layer: `Linear(64, 10)`
  - **Output Activation: None (linear regression head)**
- **Loss Function:** Mean Squared Error (MSELoss).
- **Optimizer:** Adam (Default learning rate: 1e-3).

## 3. Data Schema & Preprocessing
**Inputs (`[Batch, 15]` tensor):**
- 10 noisy signal samples.
- 4-dimensional categorical one-hot vector (representing the discrete frequency $C$).
- 1 continuous float representing the amplitude noise percentage ($\sigma$).

**Outputs (`[Batch, 10]` tensor):**
- 10 clean, denoised signal samples representing the target ground truth.

**Preprocessing Assumptions:**
- All continuous signal values (both input samples and targets) must be standardized to zero mean and unit variance per dataset split prior to network ingestion to ensure stable gradient flow.

## 4. Constraints, Limitations, and Alternatives
**Hard Constraints:**
- Must be implemented in PyTorch under the strict 150-line file limit. This constraint prioritizes minimal model definition, discourages auxiliary abstraction layers, and requires consolidating the model class and forward pass efficiently.

**Limitations:**
- The flattening of the context window removes inherent temporal relationships. The MLP must infer the time sequence purely from the fixed input weights, making it rigid compared to recurrent architectures.

**Alternatives Considered:**
- Linear Regression was considered but rejected because it lacks the deep, non-linear capacity required to map the complex distortions of combined amplitude and phase noise back to a pure sine wave.

## 5. Success Criteria & Test Scenarios
**Success Criteria:**
- **Execution:** Successfully compiles and processes the `[Batch, 15]` input tensors into `[Batch, 10]` outputs without dimensionality errors.
- **KPI:** Must achieve a $\ge 30\%$ MSE reduction compared to the baseline MSE of the raw noisy input.
- **Comparative Performance:** Expected to underperform sequence models (e.g., RNN, LSTM) in final MSE due to its lack of temporal memory.

**Test Scenarios:**
- *Zero Noise Baseline:* When fed a signal with $\sigma = 0$, the network should approximate the identity mapping with negligible reconstruction error (MSE < $1e^{-4}$).
- *Dimensionality Robustness:* Verify the network gracefully handles varying batch sizes (e.g., Batch=1 vs. Batch=32) without shape mismatches.