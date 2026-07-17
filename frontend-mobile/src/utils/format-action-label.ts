/**
 * Backend action labels (YOWOv2/AVA, e.g. "carry/hold (an object)",
 * "bend/bow(at the waist)") are written for a label map, not a UI. This turns
 * them into a clean display string without touching the underlying value used
 * for MET lookups - e.g. "hand wave" -> "Hand Wave", "carry/hold (an object)"
 * -> "Carry / Hold".
 */
export function formatActionLabel(label: string | null | undefined): string {
  if (!label) return 'Auto-Detect';

  const withoutParens = label.replace(/\s*\([^)]*\)/g, '').trim();
  const spaced = withoutParens.replace(/\//g, ' / ');

  return spaced
    .split(' ')
    .filter(Boolean)
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}
