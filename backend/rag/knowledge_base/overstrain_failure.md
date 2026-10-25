# Overstrain Failure - Troubleshooting Manual
Document status: SYNTHETIC (authored for this portfolio project). Operating limits mirror the rules used to generate the AI4I 2020 dataset.
Failure type: Overstrain Failure

## Overview
A worn tool needs more torque to cut. The product of tool_wear and torque measures the cumulative mechanical load on the tool, holder and spindle. When it exceeds the limit for the product quality variant, the machine is overstrained.

## Operating Limit
Overstrain failure occurs when tool_wear x torque exceeds 11,000 minNm for quality L, 12,000 minNm for quality M, and 13,000 minNm for quality H. The machine type must be known to apply the correct limit.

## Sensor Signature
High torque together with moderately high tool wear. Torque alone or tool wear alone may look acceptable while their product exceeds the limit.

## Diagnostic Checks
1. Look up the machine type (L, M or H) and its limit.
2. Compute tool_wear x torque and compare it with that limit.
3. Inspect the tool and holder for deformation, and the spindle bearings for play.
4. Check workpiece clamping and fixture rigidity.
5. Review feed rate and depth of cut settings.

## Corrective Actions
Reduce depth of cut or feed, replace the tool earlier, and correct clamping. Inspect the spindle if the overload recurs.

## Prevention
Track the tool_wear x torque product as a trend and replace tools before it approaches the limit.
