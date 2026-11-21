| Run | Concurrent AUROC | Anticipation 2 s [95% CI] | Anticipation 4 s [95% CI] | Far-field (far_rows_0_7) antic. 2 s | WARNING+ raised | False-alarm runs | ms/window |
| --- | --- | --- | --- | --- | --- | --- | --- |
| framediff | 0.874 | 0.535 [0.39, 0.67] | 0.549 [0.40, 0.70] | 0.593 | 1/9 | 0 | 7 |
| vitb_persistence | 0.830 | 0.508 [0.32, 0.69] | 0.480 [0.37, 0.61] | 0.652 | 3/9 | 1 | 40 |
| vitb_ridge_inclip | 0.954 | 0.802 [0.77, 0.83] | 0.593 [0.54, 0.63] | 0.783 | 7/9 | 8 | 40 |
| vitb_ridge_crossseason | 0.867 | 0.669 [0.55, 0.79] | 0.592 [0.49, 0.68] | 0.668 | 1/9 | 1 | 40 |
| vitg_predictor | 0.882 | 0.687 [0.55, 0.84] | 0.642 [0.48, 0.82] | 0.708 | 3/9 | 2 | 268 |

| Run | Pooling | Concurrent AUROC | Anticipation 2 s [95% CI] | Anticipation 4 s [95% CI] | Windows with region |
| --- | --- | --- | --- | --- | --- |
| framediff | frame p95 | 0.874 | 0.535 [0.39, 0.67] | 0.549 [0.40, 0.70] | all |
| framediff | pair-local | 0.867 | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 153 |
| framediff | actors | 0.883 | 0.524 [0.40, 0.64] | 0.527 [0.40, 0.64] | 567 |
| vitb_persistence | frame p95 | 0.830 | 0.508 [0.32, 0.69] | 0.480 [0.37, 0.61] | all |
| vitb_persistence | pair-local | 0.866 | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 153 |
| vitb_persistence | actors | 0.891 | 0.511 [0.38, 0.62] | 0.522 [0.45, 0.58] | 567 |
| vitb_ridge_inclip | frame p95 | 0.954 | 0.802 [0.77, 0.83] | 0.593 [0.54, 0.63] | all |
| vitb_ridge_inclip | pair-local | 0.861 | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 153 |
| vitb_ridge_inclip | actors | 0.936 | 0.593 [0.54, 0.65] | 0.610 [0.57, 0.68] | 567 |
| vitb_ridge_crossseason | frame p95 | 0.867 | 0.669 [0.55, 0.79] | 0.592 [0.49, 0.68] | all |
| vitb_ridge_crossseason | pair-local | 0.866 | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 153 |
| vitb_ridge_crossseason | actors | 0.887 | 0.572 [0.46, 0.68] | 0.586 [0.49, 0.68] | 567 |
| vitg_predictor | frame p95 | 0.882 | 0.687 [0.55, 0.84] | 0.642 [0.48, 0.82] | all |
| vitg_predictor | pair-local | 0.864 | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 153 |
| vitg_predictor | actors | 0.864 | 0.625 [0.49, 0.75] | 0.597 [0.48, 0.69] | 567 |
