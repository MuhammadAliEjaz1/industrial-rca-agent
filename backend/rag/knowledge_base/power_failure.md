# Power Failure - Troubleshooting Manual
Document status: SYNTHETIC (authored for this portfolio project). Operating limits mirror the rules used to generate the AI4I 2020 dataset.
Failure type: Power Failure

## Overview
Mechanical power equals torque multiplied by angular velocity: power in watts = torque (Nm) x rotational speed (rpm) x 2 x pi / 60. The drive operates safely inside a defined power window.

## Operating Limit
Power failure occurs when the derived power falls below 3,500 W or rises above 9,000 W.

## Sensor Signature
Derived power outside the 3,500-9,000 W window. High power usually comes from high torque combined with high speed. Low power comes from low torque, low speed, or both.

## Diagnostic Checks
1. Recompute power from the logged torque and rotational speed and compare with both limits.
2. Determine whether the excursion is high (overload) or low (loss of drive).
3. For high power, inspect the load: workpiece, feed rate and depth of cut. Check motor and drive temperature and fault logs.
4. For low power, check for belt or coupling slippage, supply voltage sag and drive faults.
5. Verify torque and speed sensor calibration.

## Corrective Actions
Reduce feed rate or depth of cut for overload. Repair or tighten drive components and correct supply issues for low power.

## Prevention
Trend derived power per machine, calibrate torque sensors on schedule, and inspect drive components during quarterly maintenance.
