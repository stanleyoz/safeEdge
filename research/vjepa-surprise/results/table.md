| Run | Concurrent AUROC | Anticipation 2 s [95% CI] | Anticipation 4 s [95% CI] | Far-field (far_rows_0_7) antic. 2 s | WARNING+ raised | False-alarm runs | ms/window |
| --- | --- | --- | --- | --- | --- | --- | --- |
| framediff | 0.853 | 0.483 [0.34, 0.63] | 0.534 [0.39, 0.69] | 0.552 | 0/8 | 2 | 7 |
| vitb_persistence | 0.818 | 0.454 [0.31, 0.61] | 0.490 [0.41, 0.60] | 0.585 | 2/8 | 3 | 40 |
| vitb_ridge_inclip | 0.955 | 0.575 [0.53, 0.62] | 0.583 [0.53, 0.62] | 0.619 | 5/8 | 9 | 40 |
| vitb_ridge_crossseason | 0.850 | 0.619 [0.50, 0.74] | 0.596 [0.49, 0.69] | 0.626 | 2/8 | 2 | 40 |
| vitg_predictor | 0.888 | 0.697 [0.55, 0.84] | 0.647 [0.47, 0.83] | 0.698 | 2/8 | 2 | 268 |

| Run | Pooling | Concurrent AUROC | Anticipation 2 s [95% CI] | Anticipation 4 s [95% CI] | Windows with region |
| --- | --- | --- | --- | --- | --- |
| framediff | frame p95 | 0.853 | 0.483 [0.34, 0.63] | 0.534 [0.39, 0.69] | all |
| framediff | pair-local | 0.894 | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 185 |
| framediff | actors | 0.893 | 0.484 [0.43, 0.54] | 0.526 [0.46, 0.61] | 237 |
| vitb_persistence | frame p95 | 0.818 | 0.454 [0.31, 0.61] | 0.490 [0.41, 0.60] | all |
| vitb_persistence | pair-local | 0.894 | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 185 |
| vitb_persistence | actors | 0.908 | 0.486 [0.43, 0.54] | 0.528 [0.46, 0.61] | 237 |
| vitb_ridge_inclip | frame p95 | 0.955 | 0.575 [0.53, 0.62] | 0.583 [0.53, 0.62] | all |
| vitb_ridge_inclip | pair-local | 0.894 | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 185 |
| vitb_ridge_inclip | actors | 0.926 | 0.479 [0.45, 0.50] | 0.478 [0.44, 0.50] | 237 |
| vitb_ridge_crossseason | frame p95 | 0.850 | 0.619 [0.50, 0.74] | 0.596 [0.49, 0.69] | all |
| vitb_ridge_crossseason | pair-local | 0.894 | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 185 |
| vitb_ridge_crossseason | actors | 0.907 | 0.486 [0.43, 0.54] | 0.529 [0.46, 0.61] | 237 |
| vitg_predictor | frame p95 | 0.888 | 0.697 [0.55, 0.84] | 0.647 [0.47, 0.83] | all |
| vitg_predictor | pair-local | 0.894 | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 185 |
| vitg_predictor | actors | 0.897 | 0.486 [0.43, 0.54] | 0.530 [0.46, 0.62] | 237 |
