| Run | Concurrent AUROC | Anticipation 2 s [95% CI] | Anticipation 4 s [95% CI] | Far-field (far_rows_0_7) antic. 2 s | WARNING+ raised | False-alarm runs | ms/window |
| --- | --- | --- | --- | --- | --- | --- | --- |
| framediff | 0.835 | 0.903 [0.81, 0.97] | 0.838 [0.71, 0.95] | 0.887 | 3/5 | 4 | 7 |
| vitb_persistence | 0.707 | 0.847 [0.73, 0.96] | 0.775 [0.61, 0.94] | 0.854 | 1/5 | 5 | 40 |
| vitb_ridge_inclip | 0.975 | n/a | n/a | - | 3/5 | 6 | 40 |
| vitb_ridge_crossseason | 0.754 | 0.868 [0.75, 0.97] | 0.830 [0.68, 0.96] | 0.871 | 1/5 | 5 | 40 |
| vitg_predictor | 0.808 | 0.870 [0.76, 0.96] | 0.860 [0.74, 0.96] | 0.850 | 1/5 | 6 | 268 |

| Run | Pooling | Concurrent AUROC | Anticipation 2 s [95% CI] | Anticipation 4 s [95% CI] | Windows with region |
| --- | --- | --- | --- | --- | --- |
| framediff | frame p95 | 0.835 | 0.903 [0.81, 0.97] | 0.838 [0.71, 0.95] | all |
| framediff | pair-local | 0.813 | 0.897 [0.81, 0.96] | 0.894 [0.83, 0.95] | 153 |
| framediff | actors | 0.794 | 0.873 [0.77, 0.95] | 0.853 [0.74, 0.95] | 243 |
| vitb_persistence | frame p95 | 0.707 | 0.847 [0.73, 0.96] | 0.775 [0.61, 0.94] | all |
| vitb_persistence | pair-local | 0.801 | 0.895 [0.81, 0.96] | 0.887 [0.82, 0.96] | 153 |
| vitb_persistence | actors | 0.764 | 0.846 [0.74, 0.93] | 0.825 [0.70, 0.94] | 243 |
| vitb_ridge_inclip | frame p95 | 0.975 | n/a | n/a | all |
| vitb_ridge_inclip | pair-local | 0.891 | n/a | n/a | 153 |
| vitb_ridge_inclip | actors | 0.911 | n/a | n/a | 243 |
| vitb_ridge_crossseason | frame p95 | 0.754 | 0.868 [0.75, 0.97] | 0.830 [0.68, 0.96] | all |
| vitb_ridge_crossseason | pair-local | 0.797 | 0.893 [0.81, 0.96] | 0.888 [0.82, 0.96] | 153 |
| vitb_ridge_crossseason | actors | 0.760 | 0.845 [0.74, 0.93] | 0.825 [0.70, 0.94] | 243 |
| vitg_predictor | frame p95 | 0.808 | 0.870 [0.76, 0.96] | 0.860 [0.74, 0.96] | all |
| vitg_predictor | pair-local | 0.799 | 0.873 [0.78, 0.94] | 0.870 [0.81, 0.93] | 153 |
| vitg_predictor | actors | 0.770 | 0.809 [0.69, 0.90] | 0.796 [0.68, 0.90] | 243 |
