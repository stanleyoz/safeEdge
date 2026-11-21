| Run | Concurrent AUROC | Anticipation 2 s [95% CI] | Anticipation 4 s [95% CI] | Far-field (far_rows_0_7) antic. 2 s | WARNING+ raised | False-alarm runs | ms/window |
| --- | --- | --- | --- | --- | --- | --- | --- |
| framediff | 0.879 | 0.554 [0.41, 0.69] | 0.559 [0.41, 0.71] | 0.610 | 0/9 | 0 | 7 |
| vitb_persistence | 0.838 | 0.514 [0.34, 0.69] | 0.484 [0.37, 0.61] | 0.646 | 2/9 | 1 | 40 |
| vitb_ridge_inclip | 0.963 | 0.835 [0.81, 0.91] | 0.657 [0.61, 0.91] | 0.853 | 7/9 | 9 | 40 |
| vitb_ridge_crossseason | 0.875 | 0.670 [0.55, 0.79] | 0.595 [0.49, 0.68] | 0.671 | 0/9 | 1 | 40 |
| vitg_predictor | 0.888 | 0.689 [0.56, 0.84] | 0.645 [0.49, 0.81] | 0.704 | 2/9 | 2 | 269 |

| Run | Pooling | Concurrent AUROC | Anticipation 2 s [95% CI] | Anticipation 4 s [95% CI] | Windows with region |
| --- | --- | --- | --- | --- | --- |
| framediff | frame p95 | 0.879 | 0.554 [0.41, 0.69] | 0.559 [0.41, 0.71] | all |
| framediff | pair-local | 0.867 | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 153 |
| framediff | actors | 0.890 | 0.561 [0.43, 0.67] | 0.572 [0.43, 0.70] | 564 |
| vitb_persistence | frame p95 | 0.838 | 0.514 [0.34, 0.69] | 0.484 [0.37, 0.61] | all |
| vitb_persistence | pair-local | 0.866 | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 153 |
| vitb_persistence | actors | 0.896 | 0.540 [0.34, 0.68] | 0.508 [0.40, 0.61] | 564 |
| vitb_ridge_inclip | frame p95 | 0.963 | 0.835 [0.81, 0.91] | 0.657 [0.61, 0.91] | all |
| vitb_ridge_inclip | pair-local | 0.861 | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 153 |
| vitb_ridge_inclip | actors | 0.948 | 0.742 [0.69, 1.00] | 0.611 [0.56, 1.00] | 564 |
| vitb_ridge_crossseason | frame p95 | 0.875 | 0.670 [0.55, 0.79] | 0.595 [0.49, 0.68] | all |
| vitb_ridge_crossseason | pair-local | 0.866 | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 153 |
| vitb_ridge_crossseason | actors | 0.896 | 0.584 [0.43, 0.69] | 0.565 [0.44, 0.68] | 564 |
| vitg_predictor | frame p95 | 0.888 | 0.689 [0.56, 0.84] | 0.645 [0.49, 0.81] | all |
| vitg_predictor | pair-local | 0.864 | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 153 |
| vitg_predictor | actors | 0.859 | 0.553 [0.44, 0.66] | 0.574 [0.45, 0.70] | 564 |
