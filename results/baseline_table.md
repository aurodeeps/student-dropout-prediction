# Baseline table (validation set; CV on training set)

| feature_set | model | macro_F1 | accuracy | recall_Dropout | recall_Enrolled | recall_Graduate | cv_macro_F1_mean | cv_macro_F1_sd |
|---|---|---|---|---|---|---|---|---|
| early | Majority class | 0.222 | 0.500 | 0.000 | 0.000 | 1.000 | 0.222 | 0.000 |
| early | Stratified random (seed 42) | 0.313 | 0.373 | 0.319 | 0.126 | 0.497 | 0.333 | 0.020 |
| early | Logistic regression (C=1) | 0.651 | 0.736 | 0.714 | 0.319 | 0.901 | 0.659 | 0.011 |
| early | Logistic regression (tuned, C=1) | 0.651 | 0.736 | 0.714 | 0.319 | 0.901 | 0.659 | 0.011 |
| early | Decision tree (depth 5) | 0.549 | 0.703 | 0.690 | 0.084 | 0.934 | 0.576 | 0.026 |
| full | Majority class | 0.222 | 0.500 | 0.000 | 0.000 | 1.000 | 0.222 | 0.000 |
| full | Stratified random (seed 42) | 0.313 | 0.373 | 0.319 | 0.126 | 0.497 | 0.333 | 0.020 |
| full | Logistic regression (C=1) | 0.705 | 0.773 | 0.751 | 0.420 | 0.913 | 0.697 | 0.015 |
| full | Logistic regression (tuned, C=1) | 0.705 | 0.773 | 0.751 | 0.420 | 0.913 | 0.697 | 0.015 |
| full | Decision tree (depth 5) | 0.602 | 0.720 | 0.634 | 0.193 | 0.964 | 0.639 | 0.016 |
