'use client';

import { JOB_STATUS, FALLBACK_STATUS } from '@/lib/jobStatus';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface Props {
  status: string;
  progress?: number | null;
  showLabel?: boolean;
  className?: string;
}

export function StatusBadge({ status, progress, showLabel = true, className }: Props) {
  const style = JOB_STATUS[status as keyof typeof JOB_STATUS] ?? FALLBACK_STATUS;
  const Icon = style.icon;

  return (
    <Badge variant="outline" className={cn('gap-1 text-[10px]', style.class, className)}>
      <Icon className={cn('h-3 w-3', status === 'running' && 'animate-spin')} />
      {showLabel && (status === 'running' && progress != null ? `${progress}%` : style.label)}
    </Badge>
  );
}
