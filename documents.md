# MOPSO-Optimized CNN-LSTM-GRU Pipeline Formulas

This document summarizes the formulas used in the MOPSO-optimized hybrid CNN-LSTM-GRU pipeline, following the flow implemented in the root training pipeline and the shared model/preprocessing code.

## 1) Data Loading and Label Encoding

**Stage encoding**

$y_t = \text{map}(\text{Stage}_t)$, with

$\text{map} = \{\text{Awake}:0,\ \text{Light}:1,\ \text{Deep}:2,\ \text{REM}:3\}$.

**Symbol meanings**
- $t$: time index in the raw sequence.
- $y_t$: encoded sleep stage label at time $t$.

## 2) Smoothing (Selected Channels)

For each selected channel $x_t$ (e.g., Ambient_Temp, Humidity, Light_Lux), apply a moving average:

$\tilde{x}_t = \frac{1}{K} \sum_{i=0}^{K-1} x_{t-i}$.

**Symbol meanings**
- $K$: smoothing kernel size (3 in code).
- $\tilde{x}_t$: smoothed value.

## 3) Temporal Feature Engineering

**Deltas**

$\Delta x_t = x_t - x_{t-1}$.

**Rolling mean and std (window = 5)**

$\mu_t = \frac{1}{5} \sum_{i=0}^{4} x_{t-i}$,

$\sigma_t = \sqrt{\frac{1}{5} \sum_{i=0}^{4} (x_{t-i} - \mu_t)^2}$.

**Symbol meanings**
- $x_t$: base feature at time $t$.
- $\Delta x_t$: delta feature.
- $\mu_t, \sigma_t$: rolling mean and std.

## 4) Robust Scaling (Train Split Only)

For each feature $x$:

$x' = \frac{x - \text{median}(x)}{\text{IQR}(x)}$,

where $\text{IQR}(x) = Q_3 - Q_1$.

**Symbol meanings**
- $Q_1, Q_3$: first and third quartiles.
- $x'$: scaled feature.

## 5) User-Stratified Split

Users are split into train/val/test without leakage. Let $U$ be the set of user IDs. The split partitions $U$ into $U_{tr}, U_{va}, U_{te}$ with fixed ratios, and each split includes all samples for its users.

## 6) Sliding-Window Segmentation

Given a sequence of length $T$, window size $W$, and stride $S$:

$X^{(n)} = [x_{s_n}, x_{s_n+1}, \dots, x_{s_n+W-1}]$,

$y^{(n)} = y_{s_n + \lfloor W/2 \rfloor}$,

where $s_n = nS$.

**Symbol meanings**
- $X^{(n)} \in \mathbb{R}^{W \times F}$: windowed feature matrix.
- $F$: number of features.
- Center-of-window labeling preserves minority stages.

## 7) Minority Oversampling (Training Only)

Let $c$ be a class with count $N_c$, and $N_{max}$ the largest class count. Target count is $N_{target} = \lfloor r \cdot N_{max} \rfloor$, with $r=0.6$.

If $N_c < N_{target}$, then sample with replacement:

X_c^{\text{extra}} = \{ X_i \}_{i \sim \text{Uniform}(1, N_c)} \ \text{for} \ N_{target} - N_c \ \text{samples}.

## 8) Class Weights (Balanced + Minority Boost)

Balanced weights:

$w_c^{(bal)} = \frac{N}{C \cdot N_c}$,

and with minority boost factor $\beta$ (1.5 in code):

$w_c = \begin{cases}
\beta \cdot w_c^{(bal)} & \text{if } w_c^{(bal)} > 1 \\
 w_c^{(bal)} & \text{otherwise}
\end{cases}$.

**Symbol meanings**
- $N$: total samples, $C$: number of classes.
- $N_c$: count for class $c$.

## 9) CNN Encoder

Each Conv1D block (with L2 regularization) is:

$z = \text{ReLU}(\text{BN}(\text{Conv1D}(x)))$.

**Residual block**

$z_1 = \text{ConvBlock}(x)$,

$z_2 = \text{ConvBlock}(z_1)$,

$y = z_2 + \text{Proj}(x)$.

**Squeeze-and-Excite (channel attention)**

$s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GAP}(x)))$,

$y = x \odot s$.

**Symbol meanings**
- BN: batch normalization.
- GAP: global average pooling over time.
- $\odot$: channel-wise multiplication.

## 10) BiGRU and BiLSTM Branches

Given CNN feature sequence $H \in \mathbb{R}^{T' \times D}$:

$G = \text{BiGRU}(H)$,

$L = \text{BiLSTM}(H)$,

$g = \text{GAP}(G)$,

$l = \text{GAP}(L)$.

**Symbol meanings**
- $T'$: time length after CNN pooling.
- $D$: feature dimension.

## 11) CNN Global Context Branch

$c = \text{GAP}(H)$.

## 12) Feature Fusion and Classification Head

Merge branches:

$m = [c; g; l]$ (concatenation).

Dense head:

$h_1 = \text{ReLU}(W_1 m + b_1)$,

$h_2 = \text{ReLU}(W_2 h_1 + b_2)$,

$\hat{y} = \text{softmax}(W_3 h_2 + b_3)$.

## 13) Focal Loss (Class Imbalance)

For a sample with true class $t$ and predicted probability $p_t$:

$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$.

**Symbol meanings**
- $\alpha_t$: class weight for class $t$.
- $\gamma$: focal focusing parameter.

## 14) MOPSO Hyperparameter Optimization

### 14.1 Position Decoding

Each particle position $p \in [0,1]^d$ is decoded into hyperparameters:

$\text{dropout} = 0.10 + 0.40 \cdot p_5$,

$\text{learning\_rate} = 10^{-4 + 2 p_6}$,

discrete dimensions use rounded index selection over allowed values.

### 14.2 Objectives

For candidate model $\theta$:

$f_1(\theta) = 1 - \text{Acc}_{val}$,

$f_2(\theta) = \frac{\#\{\hat{y}=0 \land y>0\}}{\#\{y>0\}}\ \ \text{(false alarm rate)},

$f_3(\theta) = \log_{10}(\text{ParamCount}(\theta)).

### 14.3 Velocity and Position Update

$v \leftarrow w v + c_1 r_1 (p_{best} - p) + c_2 r_2 (g_{best} - p)$,

$p \leftarrow \text{clip}(p + v, 0, 1)$.

**Symbol meanings**
- $w$: inertia weight, $c_1, c_2$: cognitive/social coefficients.
- $r_1, r_2$: uniform random vectors.
- $p_{best}$: particle best, $g_{best}$: guide from Pareto archive.

## 15) End-to-End Pipeline Flow (Formula View)

1. **Load + Encode**: $\{x_t, \text{Stage}_t\} \to y_t$.
2. **Smooth + Engineer**: $x_t \to \tilde{x}_t, \Delta x_t, \mu_t, \sigma_t$.
3. **Split by User**: $U \to (U_{tr}, U_{va}, U_{te})$.
4. **Scale**: $x \to x' = (x - \text{median})/\text{IQR}$.
5. **Window**: $X^{(n)}, y^{(n)}$ via $(W, S)$.
6. **Balance**: oversample + class weights $w_c$.
7. **CNN Encoder**: $H = \text{CNN}(X^{(n)})$.
8. **Sequence Branches**: $G=\text{BiGRU}(H)$, $L=\text{BiLSTM}(H)$.
9. **Pool + Merge**: $m=[\text{GAP}(H);\text{GAP}(G);\text{GAP}(L)]$.
10. **Predict**: $\hat{y} = \text{softmax}(\text{Dense}(m))$.
11. **Loss**: $\text{FL}(p_t)$.
12. **MOPSO**: minimize $(f_1, f_2, f_3)$ to select hyperparameters.
