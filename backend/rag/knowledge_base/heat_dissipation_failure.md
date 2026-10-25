# Heat Dissipation Failure - Troubleshooting Manual
Document status: SYNTHETIC (authored for this portfolio project). Operating limits mirror the rules used to generate the AI4I 2020 dataset.
Failure type: Heat Dissipation Failure

## Overview
The process must shed heat into the surrounding air. When the temperature difference between process and air is small and the spindle turns slowly, heat cannot dissipate fast enough and the machine fails.

## Operating Limit
Heat dissipation failure occurs when the temperature difference (process temperature minus air temperature) is below 8.6 K and the rotational speed is below 1380 rpm at the same time. Both conditions must hold together.

## Sensor Signature
A narrow temperature gap (temp_diff under 8.6 K) combined with low rotational speed. The absolute process temperature can look normal, because the gap matters and not the absolute value.

## Diagnostic Checks
1. Compute temp_diff from the logged process and air temperature and compare it with 8.6 K.
2. Check rotational speed against 1380 rpm.
3. Inspect cooling fans, coolant level and flow, and heat exchanger cleanliness.
4. Check shop ambient temperature. A hot floor raises air temperature and closes the gap.
5. Verify calibration of both temperature sensors, since a drifting sensor can fake a narrow gap.

## Corrective Actions
Restore coolant flow, clean fans and filters, improve ventilation, and raise rotational speed where the process allows. Recalibrate suspect temperature sensors.

## Prevention
Schedule coolant system inspections, monitor temp_diff as a trend, and avoid long low-speed runs in hot conditions.
