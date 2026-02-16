

## 🔹 Technical Note (for Methods / Reproducibility)

> The models output class probabilities using a softmax layer. Final class predictions were obtained by selecting the class with the highest probability (argmax). For binary classification, this is equivalent to assigning the positive class when its predicted probability exceeds 0.5. Sensitivity and other confusion-matrix–based metrics were derived from these predictions. AUROC was computed directly from the predicted probabilities and does not depend on a fixed threshold.

This clearly distinguishes:

* **Decision metrics** vs
* **Ranking metrics**

---

## 🔹 How the threshold is implemented?

### 1️⃣ Model outputs logits

```python
logits = model(x)
```

### 2️⃣ Convert logits → probabilities

```python
probs = torch.softmax(logits, dim=1)
```

This gives:

* `probs[:, 0]` → P(Normal)
* `probs[:, 1]` → P(AF)

---

### 3️⃣ Class decision via `argmax`

```python
preds = logits.argmax(dim=1)
```

This selects:

```text
pred = 1  if  P(AF) > P(Normal)
pred = 0  otherwise
```

Because:
[
P(AF) + P(Normal) = 1
]

This is mathematically equivalent to:

```python
preds = (probs[:, 1] >= 0.5)
```

➡️ **This is where the implicit threshold = 0.5 is applied**

---

### 4️⃣ Sensitivity (Recall) calculation

```python
recall = tp / (tp + fn + 1e-12)
```

Where:

* `tp` and `fn` are computed from `preds`
* Therefore, Sensitivity depends on the implicit 0.5 threshold

---

### 5️⃣ AUROC calculation (threshold-free)

```python
roc_auc_score(y_true, y_score)
```

Where:

```python
y_score = probs[:, 1]
```

➡️ AUROC uses **continuous probabilities**, not class decisions.

