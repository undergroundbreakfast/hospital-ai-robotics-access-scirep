# System-Membership Sensitivity

## Balance
| analysis_level | measure                                          | n    | system_member_n | non_system_n | system_member_mean_pct | non_system_mean_pct | difference_pct_points | smd   |
| -------------- | ------------------------------------------------ | ---- | --------------- | ------------ | ---------------------- | ------------------- | --------------------- | ----- |
| Hospital       | MO11 staff-scheduling AI                         | 2301 | 1690            | 611          | 24.438                 | 4.910               | 19.528                | 0.574 |
| Hospital       | MO14 routine-task AI                             | 2301 | 1690            | 611          | 34.438                 | 9.984               | 24.454                | 0.616 |
| Hospital       | MO21 in-hospital robotics                        | 3045 | 2197            | 848          | 58.716                 | 33.608              | 25.108                | 0.520 |
| County         | System-member share of AHA adjusted patient days | 3080 | 417             | 2663         | 82.830                 | 43.028              | 39.802                | 1.024 |
| County         | System-member share of AHA hospitals             | 3080 | 417             | 2663         | 80.125                 | 43.032              | 37.093                | 0.985 |
| County         | AHA hospitals in county                          | 3080 | 417             | 2663         | 5.887                  | 1.350               | 4.537                 | 0.696 |

## Linear-Nuisance AIPW Sensitivity
| outcome             | specification                                  | n    | treated_n | control_n | estimate | se      | ci_lower | ci_upper | propensity_min | propensity_max | propensity_mean |
| ------------------- | ---------------------------------------------- | ---- | --------- | --------- | -------- | ------- | -------- | -------- | -------------- | -------------- | --------------- |
| dv21_ypll           | Base covariates only                           | 2899 | 415       | 2484      | -438.246 | 156.530 | -745.045 | -131.448 | 0.010          | 0.961          | 0.143           |
| dv21_ypll           | Base covariates + system-member capacity share | 2899 | 415       | 2484      | -355.098 | 153.673 | -656.297 | -53.899  | 0.010          | 0.937          | 0.145           |
| ct6_hospital_deaths | Base covariates only                           | 2897 | 415       | 2482      | -3.724   | 6.504   | -16.472  | 9.024    | 0.010          | 0.961          | 0.143           |
| ct6_hospital_deaths | Base covariates + system-member capacity share | 2897 | 415       | 2482      | 0.318    | 5.403   | -10.273  | 10.908   | 0.010          | 0.937          | 0.145           |
