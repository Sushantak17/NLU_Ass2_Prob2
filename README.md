# NLU Assignment 2 — Problem 2: Character-Level Name Generation Using RNN Variants

## Requirements

```bash
pip install torch
```

---

## File Structure

```
NLU_Assignment2_Prob2/
├── M25CSA035_Prob2.py     # Main script with all models and evaluation
└── training_names.txt     # 1000 LLM-generated Indian names (must be in same directory)
```

> **Important:** `training_names.txt` must be in the same directory as the script before running.

---

## How to Run

The entire assignment is contained in a **single script**. Run it as:

```bash
python M25CSA035_Prob2.py
```

The script runs end-to-end in the following order:

### Dataset & Vocabulary
Loads `training_names.txt`, builds the character vocabulary, and creates padded input-output training sequences.

### Model 1 — Vanilla RNN
Defines and trains a character-level RNN (hidden_size=128, epochs=40, lr=0.003). Generates and filters names, then computes novelty rate and diversity on 500 samples.

### Model 2 — Bidirectional LSTM (BLSTM)
Defines and trains a BiLSTM (hidden_size=128, epochs=20, lr=0.003). Generates and filters names, then computes novelty rate and diversity on 500 samples.

### Model 3 — RNN with Attention
Defines and trains an RNN with a basic attention mechanism (hidden_size=128, epochs=40, lr=0.003). Generates and filters names, then computes novelty rate and diversity on 500 samples.

### Cross-Model Comparison
Prints a summary table comparing novelty rate and diversity across all three models.

---

## Notes
- All three models use `hidden_size=128` and the Adam optimizer
- Name generation uses temperature sampling (`temperature=0.7–0.8`) for controlled randomness
- Evaluation is done on 500 generated names filtered for valid full-name structure (first + last name)
- Training the full script end-to-end takes a few minutes depending on your machine
