# Compound Failures - Troubleshooting Manual
Document status: SYNTHETIC (authored for this portfolio project). Operating limits mirror the rules used to generate the AI4I 2020 dataset.
Failure type: Compound Failure

## Overview
A single incident can involve more than one failure mechanism at the same time. Each mechanism has its own limit and each must be evaluated independently.

## Common Combinations
Power Failure with Overstrain Failure: both are driven by high torque, so derived power above 9,000 W and tool_wear x torque above the quality limit often occur together.
Heat Dissipation Failure with Overstrain Failure: low speed with a narrow temperature gap, together with high torque on a worn tool.
Tool Wear Failure with Overstrain Failure: a worn tool raises torque, which pushes the tool_wear x torque product over its limit.

## Investigation Guidance
1. Test every mechanism against its own limit. Do not stop at the first match.
2. Report all mechanisms supported by evidence, and state which are not.
3. When two mechanisms share a driver, such as high torque, treat the shared driver as the first thing to explain.
4. A classifier label that names a single cause may hide a second contributing cause.
