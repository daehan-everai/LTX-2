'use client';

import type { SourceFolder } from '@/lib/types';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { Folder, FileVideo, FileImage, Trash2, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Props {
  folder: SourceFolder;
  selected: boolean;
  onSelect: () => void;
  onRemove: () => void;
  onRefresh: () => void;
  isRefreshing: boolean;
}

export function SourceFolderCard({ folder, selected, onSelect, onRemove, onRefresh, isRefreshing }: Props) {
  const Icon = folder.mediaType === 'images' ? FileImage : FileVideo;

  return (
    <Card
      className={cn(
        'hover:border-primary/40 cursor-pointer transition-colors',
        selected && 'border-primary ring-primary/30 ring-1',
      )}
      onClick={onSelect}
    >
      <CardContent className="flex items-center gap-3 px-4 py-2.5">
        <div className="bg-muted rounded-md p-1.5">
          <Folder className="text-muted-foreground h-4 w-4" />
        </div>

        <h3 className="min-w-0 flex-1 font-mono text-sm font-medium break-all">{folder.path}</h3>

        <Badge variant="secondary" className="shrink-0 text-[10px]">
          <Icon className="mr-1 h-3 w-3" />
          {folder.fileCount}
        </Badge>

        <Button
          size="sm"
          variant="ghost"
          className="text-muted-foreground hover:text-foreground h-7 w-7 shrink-0 p-0"
          onClick={e => {
            e.stopPropagation();
            onRefresh();
          }}
          disabled={isRefreshing}
          title="Refresh folder & discover buckets"
        >
          <RefreshCw className={cn('h-3.5 w-3.5', isRefreshing && 'animate-spin')} />
        </Button>

        <ConfirmDialog
          title="Remove folder?"
          description={`This will remove "${folder.path}" from the list. Existing preprocessed data and files on disk will not be affected.`}
          confirmLabel="Remove"
          onConfirm={onRemove}
        >
          <Button
            size="sm"
            variant="ghost"
            className="text-destructive hover:text-destructive h-7 w-7 shrink-0 p-0"
            onClick={e => {
              e.stopPropagation();
            }}
            title="Remove folder"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </ConfirmDialog>
      </CardContent>
    </Card>
  );
}
