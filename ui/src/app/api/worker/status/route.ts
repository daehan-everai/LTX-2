import { NextResponse } from 'next/server';
import { getSetting } from '@/lib/settings';

const STALE_THRESHOLD_MS = 5000;

export async function GET() {
  const lastSeen = getSetting('worker_heartbeat');
  if (!lastSeen) {
    return NextResponse.json({ alive: false, lastSeen: null });
  }

  const lastSeenMs = new Date(lastSeen).getTime();
  const alive = Date.now() - lastSeenMs < STALE_THRESHOLD_MS;

  return NextResponse.json({ alive, lastSeen });
}
