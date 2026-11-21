| Run | Concurrent AUROC | Anticipation 2 s [95% CI] | Anticipation 4 s [95% CI] | Far-field (far_rows_0_7) antic. 2 s | WARNING+ raised | False-alarm runs | ms/window |
| --- | --- | --- | --- | --- | --- | --- | --- |
| framediff | 0.853 | 0.874 [0.76, 0.96] | 0.799 [0.66, 0.92] | 0.859 | 3/5 | 4 | 7 |
| vitb_persistence | 0.729 | 0.835 [0.71, 0.95] | 0.743 [0.56, 0.92] | 0.844 | 1/5 | 5 | 40 |
| vitb_ridge_inclip | 0.986 | n/a | n/a | - | 3/5 | 7 | 40 |
| vitb_ridge_crossseason | 0.770 | 0.857 [0.73, 0.97] | 0.810 [0.65, 0.94] | 0.867 | 2/5 | 5 | 40 |
| vitg_predictor | 0.820 | 0.870 [0.75, 0.96] | 0.847 [0.72, 0.95] | 0.838 | 2/5 | 5 | 268 |

| Run | Pooling | Concurrent AUROC | Anticipation 2 s [95% CI] | Anticipation 4 s [95% CI] | Windows with region |
| --- | --- | --- | --- | --- | --- |
| framediff | frame p95 | 0.853 | 0.874 [0.76, 0.96] | 0.799 [0.66, 0.92] | all |
| framediff | pair-local | 0.830 | 0.892 [0.81, 0.96] | 0.885 [0.82, 0.95] | 153 |
| framediff | actors | 0.817 | 0.871 [0.76, 0.96] | 0.837 [0.71, 0.94] | 243 |
| vitb_persistence | frame p95 | 0.729 | 0.835 [0.71, 0.95] | 0.743 [0.56, 0.92] | all |
| vitb_persistence | pair-local | 0.820 | 0.896 [0.82, 0.97] | 0.884 [0.81, 0.96] | 153 |
| vitb_persistence | actors | 0.775 | 0.857 [0.75, 0.95] | 0.823 [0.68, 0.94] | 243 |
| vitb_ridge_inclip | frame p95 | 0.986 | n/a | n/a | all |
| vitb_ridge_inclip | pair-local | 0.907 | n/a | n/a | 153 |
| vitb_ridge_inclip | actors | 0.927 | n/a | n/a | 243 |
| vitb_ridge_crossseason | frame p95 | 0.770 | 0.857 [0.73, 0.97] | 0.810 [0.65, 0.94] | all |
| vitb_ridge_crossseason | pair-local | 0.815 | 0.895 [0.82, 0.96] | 0.886 [0.81, 0.96] | 153 |
| vitb_ridge_crossseason | actors | 0.774 | 0.852 [0.74, 0.95] | 0.823 [0.68, 0.94] | 243 |
| vitg_predictor | frame p95 | 0.820 | 0.870 [0.75, 0.96] | 0.847 [0.72, 0.95] | all |
| vitg_predictor | pair-local | 0.814 | 0.875 [0.79, 0.95] | 0.873 [0.80, 0.94] | 153 |
| vitg_predictor | actors | 0.776 | 0.821 [0.71, 0.92] | 0.809 [0.68, 0.92] | 243 |
