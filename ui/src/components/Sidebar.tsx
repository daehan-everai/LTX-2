'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { Database, GraduationCap, Play, Settings, Layers, type LucideIcon } from 'lucide-react';

const NAV_ITEMS = [
  { href: '/datasets', label: 'Datasets', icon: Database },
  { href: '/training', label: 'Training', icon: GraduationCap },
  { href: '/runs', label: 'Runs', icon: Play },
] as const;

function NavLink({
  href,
  icon: Icon,
  label,
  active,
}: {
  href: string;
  icon: LucideIcon;
  label: string;
  active: boolean;
}) {
  return (
    <Link
      href={href}
      className={cn(
        'flex items-center gap-2.5 rounded-xl px-3 py-2.5 text-[0.95rem] font-medium transition-all duration-200',
        active
          ? 'ring-glow bg-sidebar-accent text-sidebar-accent-foreground shadow-[inset_0_1px_0_oklch(1_0_0_/_0.12),0_6px_18px_oklch(0.07_0.03_258_/_0.55)]'
          : 'text-sidebar-foreground/82 hover:bg-sidebar-accent/65 hover:text-sidebar-foreground',
      )}
    >
      <Icon className="h-4 w-4 shrink-0" />
      {label}
    </Link>
  );
}

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="border-sidebar-border bg-sidebar relative z-10 flex h-full w-[15.5rem] flex-col rounded-r-3xl border-r p-2 shadow-[12px_0_30px_oklch(0.08_0.03_258_/_0.52)]">
      <div className="surface-neo flex items-center gap-2.5 rounded-2xl px-4 py-4">
        <div className="ring-glow bg-primary flex h-9 w-9 items-center justify-center rounded-xl">
          <Layers className="text-primary-foreground h-4 w-4" />
        </div>
        <div>
          <h1 className="title-gradient text-lg leading-none font-bold tracking-tight">LTX Trainer</h1>
          <p className="text-sidebar-foreground/78 mt-1 text-[0.78rem]">Training UI</p>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-1 py-4">
        {NAV_ITEMS.map(({ href, label, icon }) => (
          <NavLink key={href} href={href} icon={icon} label={label} active={pathname.startsWith(href)} />
        ))}
      </nav>

      <div className="border-sidebar-border border-t px-1 py-3">
        <NavLink href="/settings" icon={Settings} label="Settings" active={pathname.startsWith('/settings')} />
      </div>
    </aside>
  );
}
