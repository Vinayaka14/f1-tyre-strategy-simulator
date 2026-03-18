# Model comparison — F1 tyre strategy simulator

## Overview
This pit stop classifier predicts when a driver should stop based on track telemetry, tyre age, and race state. The dataset comprises 65,001 rows and 12 features spanning three seasons (2022-2024). Comparing models enables us to evaluate which algorithm best handles class imbalance and captures the complex, non-linear signals of tyre degradation.

## Dataset summary
| Property | Value |
|----------|-------|
| rows | 65,001 |
| Train/Test Split | 80/20 |
| Pit Class | 5.9% |
| scale_pos_weight | 15.94 |
| Feature Count | 12 |

## Models evaluated

### XGBoost
```python
n_estimators=300, max_depth=6, learning_rate=0.05,
scale_pos_weight=15.94, random_state=42
```

### LightGBM
```python
n_estimators=300, max_depth=6, learning_rate=0.05,
scale_pos_weight=15.94, random_state=42
```

### MLP
```python
hidden_layer_sizes=(128, 64, 32), activation='relu',
max_iter=300, random_state=42, early_stopping=True
```

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| XGBoost | 0.978 | 0.748 | 0.948 | 0.836 | 0.995 |
| LightGBM| 0.977 | 0.738 | 0.954 | 0.832 | 0.995 |
| MLP     | 0.957 | 0.587 | 0.936 | 0.722 | 0.987 |

## Key findings
- We found XGBoost to be the best-performing model with an F1 score of 0.836, closely followed by LightGBM at 0.832.
- The tree-based models (XGBoost and LightGBM) handled the class imbalance excellently using native scale_pos_weight scoring.
- The MLP struggled relatively to the other two models (Precision dropped significantly to 0.587), demonstrating the tree-based approach is stronger for tabular F1 telemetry.
- Recall is generally extremely high across the board (> 0.93), ensuring very few pit stops are missed (false negatives).

## Best model selected
The selected model is XGBoost. F1 was the primary metric over accuracy due to the severe class imbalance (we care deeply about correctly predicting pit stops, not just majority-class stay-outs). It achieved an F1 of 0.836 with an incredible Recall of 0.948.

## How to reproduce
```bash
pip install -r requirements.txt
python compare_models.py
```

## Notes
Class imbalance handling is critical for achieving high recall on pit stops. The tree-based models (XGBoost, LightGBM) natively support scale_pos_weight in their cost functions, while the MLP relies on scikit-learn's compute_sample_weight to balance gradient updates during backpropagation.
