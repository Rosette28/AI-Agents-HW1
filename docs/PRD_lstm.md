# Algorithm PRD: Long Short-Term Memory (LSTM)

## 1. Theoretical Background
The Long Short-Term Memory (LSTM) network is an advanced recurrent architecture engineered to overcome the vanishing gradient problem of standard RNNs. It introduces a "Cell State" ($c_t$) alongside the hidden state ($h_t$), regulated by three distinct mathematical gates: the Forget Gate, Input Gate, and Output Gate. Through element-wise multiplication and addition, these gates dynamically decide what information to retain or discard over time. This makes the LSTM vastly superior at capturing complex, long-term dependencies within noisy continuous signals.

## 2. Architecture & Technical Specification
- **Input Dimension (Per Time Step):** 6 features (1 noisy signal sample + 4-dimensional one-hot frequency vector + 1 noise scalar).
- **Hidden & Cell Dimension:** 64 units.
- **Output Dimension (Per Time Step):** 1 feature.
- **Proposed Architecture:**
  - Recurrent Layer: `LSTM(input_size=6, hidden_size=64, batch_first=True)`
  - Output Layer: `Linear(64, 1)` applied to each time step's hidden state.
  - **Output Activation:** None (linear regression head).
- **Loss Function:** Mean Squared Error (MSELoss).
- **Optimizer:** Adam (Default learning rate: 1e-3).

## 3. Data Schema & Preprocessing
**Inputs (`[Batch, Sequence_Length=10, Features=6]` tensor):**
- Formatted identically to the RNN to allow for a shared data-loading pipeline.

**Outputs (`[Batch, Sequence_Length=10]` tensor):**
- 10 clean, denoised signal samples.

**Preprocessing Assumptions:**
- Zero mean and unit variance standardization is mandatory. Unnormalized data can push the LSTM's Sigmoid and Tanh gate activations into saturation regions, halting the learning process.

## 4. Constraints, Limitations, and Alternatives
**Hard Constraints:**
- [cite_start]Must strictly adhere to the 150-line file limit [cite: 192-194]. To achieve this without duplication, the LSTM model must inherit shared training and validation logic from a central `BaseNeuralModel` mixin.
- Both the hidden state ($h_t$) and cell state ($c_t$) must be cleanly initialized per batch.

**Limitations:**
- The gating mechanism requires significantly more tensor operations than the MLP and RNN, increasing the computational cost and training time. 
- High risk of overfitting on short sequences (10 samples) if the hidden dimension (64) is too large relative to the training data volume.

**Alternatives Considered:**
- Gated Recurrent Units (GRUs) were evaluated as a computationally cheaper alternative but rejected to ensure maximum performance and adherence to the specific architecture requested in the project brief.

## 5. Success Criteria & Test Scenarios
**Success Criteria:**
- **Execution:** Successfully manages the dual state tensors ($h_t$, $c_t$) without dimension mismatch errors.
- **KPI:** Must achieve the absolute lowest MSE across all three architectures, demonstrating superior capability in isolating pure frequency signals from high-variance noise.

**Test Scenarios:**
- *Dual-State Independence Test:* Verify that both the cell state and hidden state reset to zero between batches.
- *High Noise Robustness Test:* Test with maximum noise ($\sigma$) to definitively prove the LSTM's superior filtering compared to the MLP baseline.