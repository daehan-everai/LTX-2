'use client';

import { useState } from 'react';
import { useSettings } from '@/hooks/useSettings';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { PageHeader } from '@/components/PageHeader';
import { Save, Loader2 } from 'lucide-react';
import { toErrorMessage } from '@/lib/utils';
import type { AppSettings } from '@/lib/types';

const FIELDS: Array<{
  key: keyof AppSettings;
  label: string;
  placeholder: string;
  group: 'paths' | 'dirs';
}> = [
  {
    key: 'modelPath',
    label: 'LTX Checkpoint Path',
    placeholder: '/path/to/ltx-2.3-22b-dev.safetensors',
    group: 'paths',
  },
  {
    key: 'textEncoderPath',
    label: 'Text Encoder Path',
    placeholder: '/path/to/google/gemma-3-12b-it',
    group: 'paths',
  },
  { key: 'outputDir', label: 'Training Output Directory', placeholder: '/path/to/training/outputs', group: 'dirs' },
  {
    key: 'datasetDir',
    label: 'Dataset Directory',
    placeholder: '/path/to/preprocessed/datasets',
    group: 'dirs',
  },
  {
    key: 'scriptsDir',
    label: 'Scripts Directory',
    placeholder: '/path/to/LTX-2/packages/ltx-trainer/scripts',
    group: 'dirs',
  },
];

export default function SettingsPage() {
  const { settings, isLoading, isError, error, saveSettings, isSaving, saveError } = useSettings();

  if (isLoading || !settings) {
    return (
      <div className="flex items-center justify-center p-12">
        {isError ? (
          <p className="text-destructive text-sm">{error ? toErrorMessage(error) : 'Failed to load'}</p>
        ) : (
          <Loader2 className="text-muted-foreground h-6 w-6 animate-spin" />
        )}
      </div>
    );
  }

  return (
    <SettingsForm
      // Key resets local form state whenever the upstream settings change
      // (after save success / external refetch).
      key={JSON.stringify(settings)}
      initial={settings}
      saveSettings={saveSettings}
      isSaving={isSaving}
      saveError={saveError}
    />
  );
}

function SettingsForm({
  initial,
  saveSettings,
  isSaving,
  saveError,
}: {
  initial: AppSettings;
  saveSettings: (s: Partial<AppSettings>) => Promise<unknown>;
  isSaving: boolean;
  saveError: Error | null;
}) {
  const [local, setLocal] = useState<AppSettings>(initial);
  const [saved, setSaved] = useState(false);

  const handleSave = async () => {
    try {
      await saveSettings(local);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch {
      /* handled by saveError */
    }
  };

  const update = (key: keyof AppSettings, value: string) => {
    setLocal(prev => ({ ...prev, [key]: value }));
  };

  const paths = FIELDS.filter(f => f.group === 'paths');
  const dirs = FIELDS.filter(f => f.group === 'dirs');

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-3 md:p-4">
      <PageHeader title="Settings" subtitle="Global defaults for model paths and directories" />

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Model Paths</CardTitle>
          <CardDescription>Default model and text encoder paths used across all jobs</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {paths.map(f => (
            <FieldRow key={f.key} field={f} value={local[f.key]} onChange={v => update(f.key, v)} />
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Directories</CardTitle>
          <CardDescription>Output and scripts paths</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {dirs.map(f => (
            <FieldRow key={f.key} field={f} value={local[f.key]} onChange={v => update(f.key, v)} />
          ))}
        </CardContent>
      </Card>

      {saveError && (
        <div className="border-destructive/30 bg-destructive/10 text-destructive rounded-lg border p-3 text-sm">
          {toErrorMessage(saveError)}
        </div>
      )}

      <div className="flex justify-end">
        <Button onClick={handleSave} disabled={isSaving}>
          <Save className="mr-1.5 h-4 w-4" />
          {saved ? 'Saved!' : isSaving ? 'Saving...' : 'Save Settings'}
        </Button>
      </div>
    </div>
  );
}

function FieldRow({
  field,
  value,
  onChange,
}: {
  field: (typeof FIELDS)[number];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-2">
      <Label htmlFor={field.key}>{field.label}</Label>
      <Input
        id={field.key}
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={field.placeholder}
        className="font-mono text-xs"
      />
    </div>
  );
}
