'use client';

import { AlertDialog } from '@base-ui/react/alert-dialog';
import { cn } from '@/lib/utils';
import { Button } from './button';

interface ConfirmDialogProps {
  title: string;
  description: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: 'destructive' | 'default';
  onConfirm: () => void;
  children: React.ReactNode;
}

export function ConfirmDialog({
  title,
  description,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  variant = 'destructive',
  onConfirm,
  children,
}: ConfirmDialogProps) {
  return (
    <AlertDialog.Root>
      <AlertDialog.Trigger render={children as React.ReactElement} />
      <AlertDialog.Portal>
        <AlertDialog.Backdrop className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm transition-opacity data-[ending-style]:opacity-0 data-[starting-style]:opacity-0" />
        <AlertDialog.Popup className="border-border bg-background fixed top-1/2 left-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-2xl border p-6 shadow-2xl transition-all data-[ending-style]:scale-95 data-[ending-style]:opacity-0 data-[starting-style]:scale-95 data-[starting-style]:opacity-0">
          <AlertDialog.Title className="text-base font-semibold">{title}</AlertDialog.Title>
          <AlertDialog.Description className="text-muted-foreground mt-2 text-sm">
            {description}
          </AlertDialog.Description>
          <div className="mt-6 flex justify-end gap-2">
            <AlertDialog.Close
              render={
                <Button variant="outline" size="sm">
                  {cancelLabel}
                </Button>
              }
            />
            <AlertDialog.Close
              render={
                <Button
                  variant={variant}
                  size="sm"
                  className={cn(
                    variant === 'destructive' && 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
                  )}
                  onClick={onConfirm}
                >
                  {confirmLabel}
                </Button>
              }
            />
          </div>
        </AlertDialog.Popup>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  );
}
