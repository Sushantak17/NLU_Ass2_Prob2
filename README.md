# NLU Assignment 2 — Problem 2: Character-Level Name Generation Using RNN Variants

## Requirements

```bash
pip install torch
```

---

## File Structure

```
NLU_Assignment2_Prob2/
├── NLU_Ass2_2.ipynb       # Main notebook with all models and evaluation
└── training_names.txt     # 1000 LLM-generated Indian names (must be in same directory)
```

> **Important:** `training_names.txt` must be uploaded/placed in the same directory as the notebook before running.

---

## How to Run

The entire assignment is contained in a **single notebook**. Run the cells **top to bottom** in order.

The notebook is structured as follows:

**Dataset & Vocabulary**
Loads `training_names.txt`, builds the character vocabulary, and creates input-output training sequences.

**Model 1 — Vanilla RNN**
Defines, trains, and generates names using a character-level RNN. Computes novelty rate and diversity.

**Model 2 — Bidirectional LSTM (BLSTM)**
Defines, trains, and generates names using a BiLSTM. Computes novelty rate and diversity.

**Model 3 — RNN with Attention**
Defines, trains, and generates names using an RNN with a basic attention mechanism. Computes novelty rate and diversity.

**Cross-Model Comparison**
Prints a summary table comparing novelty rate and diversity across all three models.

**Qualitative Analysis**
Discusses realism, representative samples, and common failure modes for each model.

---

## Notes
- All three models use `hidden_size=128`, `learning_rate=0.003`, and are trained with the Adam optimizer
- Name generation uses temperature sampling (`temperature=0.7–0.8`) for controlled randomness
- Evaluation is done on 500 generated names filtered for valid full-name structure (first + last name)
