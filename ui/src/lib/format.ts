export function formatDuration(start: string, end?: string | null): string {
  const startMs = new Date(start).getTime();
  const endMs = end ? new Date(end).getTime() : Date.now();
  const diffMs = endMs - startMs;
  const mins = Math.floor(diffMs / 60000);
  const hrs = Math.floor(mins / 60);
  if (hrs > 0) return `${hrs}h ${mins % 60}m`;
  if (mins > 0) return `${mins}m`;
  return '<1m';
}
