import { cn } from '@/lib/utils';

interface Props {
  children: React.ReactNode;
  className?: string;
}

export function EmptyState({ children, className }: Props) {
  return (
    <div
      className={cn(
        'border-border text-muted-foreground rounded-lg border border-dashed p-6 text-center text-sm',
        className,
      )}
    >
      {children}
    </div>
  );
}
