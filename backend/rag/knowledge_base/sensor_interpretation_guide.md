# Sensor Interpretation Guide
Document status: SYNTHETIC (authored for this portfolio project). Operating limits mirror the rules used to generate the AI4I 2020 dataset.
Failure type: General

## Derived Quantities
temp_diff is process temperature minus air temperature. Power in watts is torque x rpm x 2 x pi / 60. The overstrain load is tool_wear x torque. Failure limits are defined on these derived quantities, so raw channels should be converted before comparison.

## Reading Trends
Compare the latest readings with the same machine's own recent history. A slow drift matters more than a single noisy reading. Steady growth in tool_wear is expected between tool changes.

## Sensor Faults Versus Real Faults
A physically implausible jump in one channel while the others stay stable suggests a sensor fault. Both temperature sensors should agree with the ambient conditions. Torque and speed sensors need scheduled calibration.

## Separating Observed Facts From Predictions
Sensor values and maintenance records are observed facts. Classifier output is a statistical prediction. A root cause is a hypothesis until sensor evidence and documentation support it.
