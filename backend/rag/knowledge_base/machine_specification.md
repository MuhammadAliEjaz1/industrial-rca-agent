# Machine Operating Envelope - Specification
Document status: SYNTHETIC (authored for this portfolio project). Operating limits mirror the rules used to generate the AI4I 2020 dataset.
Failure type: General

## Logged Channels
Air temperature in kelvin, process temperature in kelvin, rotational speed in rpm, torque in Nm, and tool wear in minutes. Typical operating values are air temperature near 300 K, process temperature near 310 K, rotational speed near 1500 rpm and torque near 40 Nm.

## Product Quality Variants
Machines run product quality L (low), M (medium) or H (high). The quality variant sets the overstrain limit.

## Failure Limits Summary
Tool wear: failure or replacement between about 200 and 240 minutes of tool wear, exact point unobservable.
Heat dissipation: temperature difference below 8.6 K together with speed below 1380 rpm.
Power: derived power below 3,500 W or above 9,000 W.
Overstrain: tool_wear x torque above 11,000 minNm (L), 12,000 minNm (M) or 13,000 minNm (H).
Random: about 0.1 percent chance per run, independent of all channels.
