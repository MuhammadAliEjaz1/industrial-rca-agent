// The documented limits from the (synthetic) technical manuals. The backend decides with the same numbers.
export const LIMITS = {
  heatGapK: 8.6,
  heatRpm: 1380,
  powerMinW: 3500,
  powerMaxW: 9000,
  wearFailMin: 200,
  overstrain: { L: 11000, M: 12000, H: 13000 },
} as const;
