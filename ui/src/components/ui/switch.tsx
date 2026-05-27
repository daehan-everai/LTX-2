'use client';

import { Switch as SwitchPrimitive } from '@base-ui/react/switch';

import { cn } from '@/lib/utils';

function Switch({
  className,
  size = 'default',
  ...props
}: SwitchPrimitive.Root.Props & {
  size?: 'sm' | 'default';
}) {
  return (
    <SwitchPrimitive.Root
      data-slot="switch"
      data-size={size}
      className={cn(
        'peer group/switch focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:border-destructive aria-invalid:ring-destructive/20 dark:aria-invalid:border-destructive/50 dark:aria-invalid:ring-destructive/40 data-checked:border-primary/60 relative inline-flex shrink-0 items-center rounded-full border border-white/14 bg-[linear-gradient(180deg,oklch(0.23_0.03_258),oklch(0.18_0.03_258))] transition-all duration-200 outline-none after:absolute after:-inset-x-3 after:-inset-y-2 focus-visible:ring-3 aria-invalid:ring-3 data-checked:bg-[linear-gradient(180deg,oklch(0.75_0.17_252),oklch(0.67_0.18_245))] data-disabled:cursor-not-allowed data-disabled:opacity-50 data-unchecked:shadow-[inset_0_1px_2px_oklch(1_0_0_/_0.05)] data-[size=default]:h-[24px] data-[size=default]:w-[44px] data-[size=sm]:h-[18px] data-[size=sm]:w-[34px]',
        className,
      )}
      {...props}
    >
      <SwitchPrimitive.Thumb
        data-slot="switch-thumb"
        className="dark:data-checked:bg-primary-foreground pointer-events-none block rounded-full bg-white shadow-[0_2px_10px_oklch(0.03_0.02_258_/_0.6)] ring-0 transition-transform duration-200 group-data-[size=default]/switch:size-[18px] group-data-[size=sm]/switch:size-[14px] group-data-[size=default]/switch:data-checked:translate-x-[23px] group-data-[size=sm]/switch:data-checked:translate-x-[17px] group-data-[size=default]/switch:data-unchecked:translate-x-[2px] group-data-[size=sm]/switch:data-unchecked:translate-x-[2px] dark:data-unchecked:bg-white"
      />
    </SwitchPrimitive.Root>
  );
}

export { Switch };
