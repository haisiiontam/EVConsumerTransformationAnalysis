# Task3 Boost Continue

- Previous: OOF `0.86185` / Final `0.8528`
- New: OOF `0.861925` / Final `0.8520`
- Scheme: `boost_B+B_concern_soft` thr=`0.5` deploy=`soft:{'M1': 0.8, 'M2': 0.15, 'M4': 0.04999999999999996}`

```
         feature_set  n_features scheme  oof_accuracy  threshold fusion_kind                                       soft_weights
         B+B_concern          22   soft      0.861925     0.5000             {"M1": 0.8, "M2": 0.15, "M4": 0.04999999999999996}
         B+B_concern          22 fusion      0.861900     0.5025      linear                                                   
         B+B_concern          22     M1      0.861850     0.5000                                                               
union_concern_travel          30   soft      0.861675     0.5925                                         {"M1": 0.2, "M2": 0.8}
union_concern_travel          30 fusion      0.861675     0.5925      linear                                                   
          B+T_travel          26   soft      0.861600     0.4925                                       {"M1": 0.45, "M2": 0.55}
          B+T_travel          26 fusion      0.861600     0.4925      linear                                                   
         B+B_concern          22     M2      0.861575     0.5000                                                               
      histish_groups          42 fusion      0.861575     0.5075      linear                                                   
      histish_groups          42   soft      0.861575     0.5075                                       {"M1": 0.65, "M2": 0.35}
      histish_groups          42     M1      0.861475     0.5000                                                               
          B+T_travel          26     M2      0.861425     0.5025                                                               
union_concern_travel          30     M2      0.861400     0.5000                                                               
          B+T_travel          26     M1      0.861375     0.5000                                                               
      histish_groups          42     M2      0.861325     0.5000                                                               
```
