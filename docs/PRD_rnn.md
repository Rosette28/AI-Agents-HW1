# Algorithm PRD: Recurrent Neural Network (RNN)

## 1. Theoretical Background
A Recurrent Neural Network (RNN) is designed for sequential data processing. Unlike the MLP, which flattens the input window, the RNN processes the signal one time-step at a time. It incorporates a "hidden state" ($h_t$) that acts as an internal memory layer, passing information from the previous time step to the current one ($h_t = f_w(x_t, h_{t-1})$). This sliding window approach allows the RNN to capture short-term temporal dependencies, making it highly effective for recognizing the high-frequency temporal patterns of our sine wave dataset.

## 2. Architecture & Technical Specification
- **Input Dimension (Per Time Step):** 6 features (1 noisy signal sample + 4-dimensional one-hot frequency vector + 1 continuous noise percentage scalar).
- **Hidden Dimension:** 64 units.
- **Output Dimension (Per Time Step):** 1 feature (the denoised signal sample).
- **Proposed Architecture:**
  - Recurrent Layer: `RNN(input_size=6, hidden_size=64, batch_first=True)`
  - Output Layer: `Linear(64, 1)` applied to each time step.
  - **Output Activation:** None (linear regression head).
- **Loss Function:** Mean Squared Error (MSELoss).
- **Optimizer:** Adam (Default learning rate: 1e-3).

## 3. Data Schema & Preprocessing
**Inputs (`[Batch, Sequence_Length=10, Features=6]` tensor):**
- The dataset generator must format the 10-sample window as a sequence of 10 distinct time steps rather than a flattened array.
- The 4-bit frequency vector and noise scalar must be appended to the noisy signal value at every single time step.

**Outputs (`[Batch, Sequence_Length=10]` tensor):**
- A sequence of 10 clean samples representing the target ground truth.

**Preprocessing Assumptions:**
- All continuous signal values must be standardized to zero mean and unit variance per dataset split to ensure stable gradient flow and prevent the exploding gradient problem common in RNNs.

## 4. Constraints, Limitations, and Alternatives
**Hard Constraints:**
- [cite_start]Must be implemented in PyTorch under the strict 150-line file limit [cite: 192-194].
- The hidden state must be explicitly initialized to zero at the start of every new batch to prevent data leakage across distinct signal sequences.

**Limitations:**
- Standard RNNs are susceptible to the vanishing gradient problem. While our context window is short (10 samples), any extended contextual dependencies might be lost over the sequence length.

**Alternatives Considered:**
- 1D Convolutional Neural Networks (CNNs) were considered for local pattern recognition but rejected because they do not natively maintain the stateful temporal memory required by the project constraints.

## 5. Success Criteria & Test Scenarios
**Success Criteria:**
- **Execution:** Successfully processes sequential `[Batch, 10, 6]` tensors into `[Batch, 10]` outputs.
- **KPI:** Must achieve a lower (better) MSE than the baseline MLP model, proving the value of temporal memory for signal reconstruction.

**Test Scenarios:**
- *Hidden State Reset Test:* Verify that the hidden state is cleared correctly between batches, ensuring the model does not carry over memory from an unrelated sine wave.
- *Sequential Alignment:* Ensure the output at time step $t=10$ correctly corresponds to the target clean sample at $t=10$.