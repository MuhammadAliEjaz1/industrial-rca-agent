# Tool Wear Failure - Troubleshooting Manual
Document status: SYNTHETIC (authored for this portfolio project). Operating limits mirror the rules used to generate the AI4I 2020 dataset.
Failure type: Tool Wear Failure

## Overview
The cutting tool degrades with cumulative use. The machine logs this as tool_wear, an accumulated-time counter in minutes. It is a counter, not a measurement of the actual cutting edge. Each tool fails or must be replaced at some point between roughly 200 and 240 minutes of accumulated wear, and the exact point differs from tool to tool and is not visible to any sensor.

## Sensor Signature
tool_wear reads in or near the 200-240 minute range. Torque may rise slightly because a dull tool needs more force, but often no other channel deviates. Many healthy readings share the same tool_wear range, so a high counter alone is a weak indicator and not proof of failure.

## Diagnostic Checks
1. Confirm the counter value and the time since the last tool change in the maintenance records.
2. Inspect the cutting edge for flank wear, chipping and built-up edge.
3. Compare current torque against the baseline torque for the same product quality at low tool wear.
4. Ask operators about vibration, chatter or poor surface finish.
5. Rule out overstrain by checking the product of tool_wear and torque against the limit for the product quality.

## Corrective Actions
Replace the tool, reset the tool_wear counter, run a first-part check, and log the replacement in the maintenance records.

## Prevention
Replace tools proactively at about 180-200 minutes instead of waiting for failure. Track wear per tool and inspect at scheduled intervals.

## Limits of Sensor-Based Diagnosis
Tool wear failure cannot be reliably separated from healthy operation using the logged counters alone. Physical inspection of the tool is required to confirm this cause.
