import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export async function parseApiError(res: Response): Promise<string> {
  const contentType = res.headers.get('content-type') ?? '';
  if (contentType.includes('application/json')) {
    try {
      const json = await res.json();
      if (typeof json?.error === 'string') return json.error;
      return JSON.stringify(json);
    } catch {
      // fall through
    }
  }
  return `Server error ${res.status}${res.statusText ? `: ${res.statusText}` : ''}`;
}

export function parseJobConfig<T = Record<string, unknown>>(raw: string): T | null {
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export function safeId(str: string | null | undefined): number | null {
  if (!str) return null;
  const n = Number(str);
  return Number.isFinite(n) ? n : null;
}

export function toErrorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
