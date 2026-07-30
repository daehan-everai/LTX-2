'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

export function Section({
  title,
  defaultOpen = true,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <Card>
      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger
          nativeButton={false}
          render={
            <CardHeader className="w-full cursor-pointer py-3 text-left select-none">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg font-semibold">{title}</CardTitle>
                <ChevronDown
                  className={cn('text-muted-foreground h-4 w-4 transition-transform', open && 'rotate-180')}
                />
              </div>
            </CardHeader>
          }
        />
        <CollapsibleContent>
          <CardContent className="space-y-4 pt-0">{children}</CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}

export function NumberField({
  label,
  value,
  onChange,
  step,
  mono,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  mono?: boolean;
}) {
  const [raw, setRaw] = useState<string>(() => String(value));

  // Keep the visible string in sync with the canonical number when it changes
  // externally (e.g. preset selection, restored config) without disturbing the
  // user's in-progress typing.
  useEffect(() => {
    if (raw === '' || raw === '-' || raw.endsWith('.') || raw.endsWith('e') || raw.endsWith('e-')) return;
    if (Number(raw) !== value) setRaw(String(value));
  }, [value, raw]);

  return (
    <div className="space-y-2">
      <Label className="text-xs">{label}</Label>
      <Input
        type="number"
        step={step}
        value={raw}
        onChange={e => {
          const next = e.target.value;
          setRaw(next);
          const n = Number(next);
          if (next !== '' && next !== '-' && Number.isFinite(n)) {
            onChange(n);
          }
        }}
        onBlur={() => {
          const n = Number(raw);
          if (!Number.isFinite(n)) setRaw(String(value));
        }}
        className={cn('text-xs', mono && 'font-mono')}
      />
    </div>
  );
}

export function TextField({
  label,
  value,
  onChange,
  placeholder,
  mono,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  mono?: boolean;
}) {
  return (
    <div className="space-y-2">
      <Label className="text-xs">{label}</Label>
      <Input
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        className={cn('text-xs', mono && 'font-mono')}
      />
    </div>
  );
}

export function SelectField({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div className="space-y-2">
      <Label className="text-xs">{label}</Label>
      <Select
        value={value}
        onValueChange={v => {
          if (v) onChange(v);
        }}
      >
        <SelectTrigger className="w-full text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map(o => (
            <SelectItem key={o.value} value={o.value}>
              {o.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

export function SwitchField({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="space-y-2">
      <Label className="text-xs">{label}</Label>
      <div className="flex h-9 items-center">
        <Switch checked={checked} onCheckedChange={onChange} />
      </div>
    </div>
  );
}

export function ListInput({
  value,
  onChange,
  placeholder,
}: {
  value: string[];
  onChange: (v: string[]) => void;
  placeholder?: string;
}) {
  // Preserve blank/in-progress lines while typing; only collapse blanks on blur
  // so the user can press Enter without losing focus on the empty new line.
  const [text, setText] = useState<string>(() => value.join('\n'));

  useEffect(() => {
    const joined = value.join('\n');
    if (joined !== text.split('\n').filter(Boolean).join('\n')) {
      setText(joined);
    }
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return (
    <Textarea
      value={text}
      onChange={e => {
        const next = e.target.value;
        setText(next);
        onChange(next.split('\n').filter(s => s.trim() !== ''));
      }}
      onBlur={() => setText(value.join('\n'))}
      placeholder={placeholder}
      rows={3}
      className="text-xs"
    />
  );
}

const VIDEO_DIM_LABELS = ['Width', 'Height', 'Frames'] as const;

export function VideoDimsField({
  value,
  onChange,
}: {
  value: [number, number, number];
  onChange: (v: [number, number, number]) => void;
}) {
  return (
    <div className="grid grid-cols-3 gap-4">
      {VIDEO_DIM_LABELS.map((label, i) => (
        <div key={label} className="space-y-2">
          <Label className="text-xs">{label}</Label>
          <Input
            type="number"
            value={value[i]}
            onChange={e => {
              const next = [...value] as [number, number, number];
              next[i] = Number(e.target.value) || 0;
              onChange(next);
            }}
            className="text-xs"
          />
        </div>
      ))}
    </div>
  );
}
