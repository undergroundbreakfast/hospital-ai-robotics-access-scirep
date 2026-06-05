# Pre-exposure CMS Quality Balance / Pretrend Analysis

## Design

This reviewer-responsive check compares pre-exposure CMS hospital-quality
performance between hospitals that subsequently reported workflow-AI adoption
in the 2023 AHA analytic year and those that did not.

Two contrasts are reported:

1. Observed 2023 adopter vs observed 2023 nonadopter.
2. Incident 2023 adopter vs stable nonadopter among hospitals reporting
   nonadoption in the 2022 AHA wave.

The incident-adopter contrast is the cleaner reviewer-facing comparison because
it uses the 2022 AHA wave to exclude hospitals that already reported adoption
before the manuscript exposure year.

## Key Interpretation

For the main SEP-1 mechanism check, hospitals that newly reported
staff-scheduling AI in 2023 did not show a statistically significant advantage
in the pre-exposure SEP-1 trajectory. The incident-adopter change comparison
estimated a 1.47-point adopter-minus-stable-nonadopter
difference in SEP-1 change from the 2019 baseline to the October 2022 CMS
public snapshot (95% CI -2.50 to 5.44;
p=0.462; SMD=0.10).

For pneumonia mortality, the latest 2022 CMS public snapshots with populated
MORT_30_PN hospital scores were January/April 2022 and used an older reporting
window. The incident routine-task AI comparison showed a
-0.45-percentage-point pre-exposure difference
(95% CI -0.89 to -0.01;
p=0.044; SMD=-0.23). Because later 2022
MORT_30_PN rows were present but unscored, this should be described as a
baseline balance check rather than a true 2022 pneumonia pretrend. This is a
warning flag rather than a reassuring result: hospitals that later reported
routine-task AI already had modestly lower pneumonia mortality in the populated
pre-exposure public-reporting window.

## Suggested Manuscript / Response Language

We added a reviewer-responsive pre-exposure balance and trend check using CMS
hospital public-reporting snapshots released in 2022 linked to AHA workflow-AI
adoption in the 2023 analytic year. For SEP-1, the closest populated
pre-exposure snapshot was the October 2022 CMS file, which reports the
January-December 2021 performance window. Among hospitals that reported no
staff-scheduling AI in the 2022 AHA wave, hospitals newly reporting adoption in
2023 did not have a statistically significant differential SEP-1 trajectory
relative to stable nonadopters from the 2019 baseline to the 2022 public
snapshot. Pneumonia mortality was evaluated as a baseline balance check because
the later 2022 CMS MORT_30_PN rows were present but not numerically scored.
These analyses do not establish parallel trends. They reduce concern that the
primary SEP-1 association is solely an artifact of visibly superior
pre-exposure CMS performance among future staff-scheduling AI adopters, while
also showing that the pneumonia mortality association should be interpreted more
cautiously because routine-task AI adopters had modestly better pre-exposure
pneumonia mortality.

## Transition Counts

| exposure             | nonadopter_2022_to_nonadopter_2023 | nonadopter_2022_to_adopter_2023 | adopter_2022_to_adopter_2023 | adopter_2022_to_nonadopter_2023 |
| -------------------- | ---------------------------------- | ------------------------------- | ---------------------------- | ------------------------------- |
| Staff-scheduling AI  | 1472                               | 82                              | 339                          | 98                              |
| Routine-task AI      | 1310                               | 95                              | 490                          | 96                              |
| In-hospital robotics | 1354                               | 72                              | 1455                         | 18                              |
