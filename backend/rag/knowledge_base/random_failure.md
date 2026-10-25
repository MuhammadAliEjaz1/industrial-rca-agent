# Random and Unexplained Failures - Troubleshooting Manual
Document status: SYNTHETIC (authored for this portfolio project). Operating limits mirror the rules used to generate the AI4I 2020 dataset.
Failure type: Random Failure

## Overview
Some failures occur with no sensor-visible cause. Historically about 0.1 percent of runs fail regardless of temperature, speed, torque or tool wear. These failures cannot be predicted from the logged sensor channels.

## Sensor Signature
None. Every logged channel is inside its normal window and no limit for tool wear, heat dissipation, power or overstrain is violated.

## Diagnostic Checks
1. Confirm that no documented limit was violated: temp_diff and speed, derived power, tool_wear x torque, and tool_wear counter.
2. Review maintenance records for recent work that might explain the fault.
3. Inspect non-instrumented causes: electrical connectors, power supply glitches, control software faults, material defects and operator error.

## Conclusion Guidance
If every mechanism has been checked and none is supported by evidence, report the failure as unexplained. Do not assign a confident root cause. Some failures are also recorded with no failure mode identified at all. These should likewise be reported as unknown.

## Corrective Actions
Restart and inspect connectors and power supply, review controller logs, and document the event so recurrence patterns can be found.
