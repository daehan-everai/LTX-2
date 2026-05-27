import { parseApiError } from './utils';

export async function apiFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(await parseApiError(res));
  return res.json();
}

const JSON_HEADERS = { 'Content-Type': 'application/json' } as const;

export function apiPost<T>(url: string, body: unknown): Promise<T> {
  return apiFetch<T>(url, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });
}

export function apiPut<T>(url: string, body: unknown): Promise<T> {
  return apiFetch<T>(url, {
    method: 'PUT',
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });
}

export function apiPatch<T>(url: string, body: unknown): Promise<T> {
  return apiFetch<T>(url, {
    method: 'PATCH',
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });
}

export function apiDelete<T>(url: string): Promise<T> {
  return apiFetch<T>(url, { method: 'DELETE' });
}
