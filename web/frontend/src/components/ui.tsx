import React from "react";

export function Panel({
  title,
  right,
  children,
  className = "",
}: {
  title?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`bg-panel border border-edge rounded-lg overflow-hidden ${className}`}
    >
      {title && (
        <div className="flex items-center justify-between px-4 py-2 border-b border-edge">
          <h3 className="text-xs uppercase tracking-widest text-muted">
            {title}
          </h3>
          {right}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}

export function Stat({
  label,
  value,
  sub,
  tone = "default",
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  tone?: "default" | "up" | "down" | "warn" | "info";
}) {
  const toneCls =
    tone === "up"
      ? "text-up"
      : tone === "down"
      ? "text-down"
      : tone === "warn"
      ? "text-warn"
      : tone === "info"
      ? "text-info"
      : "text-zinc-100";
  return (
    <div className="bg-panel2 border border-edge rounded-md px-4 py-3">
      <div className="text-[10px] uppercase tracking-widest text-muted">
        {label}
      </div>
      <div className={`text-2xl font-semibold mt-1 ${toneCls}`}>{value}</div>
      {sub && <div className="text-xs text-muted mt-1">{sub}</div>}
    </div>
  );
}

export function Badge({
  children,
  tone = "default",
}: {
  children: React.ReactNode;
  tone?: "ok" | "warn" | "crit" | "info" | "default";
}) {
  const map: Record<string, string> = {
    ok: "bg-up/15 text-up border-up/40",
    warn: "bg-warn/15 text-warn border-warn/40",
    crit: "bg-down/15 text-down border-down/40",
    info: "bg-info/15 text-info border-info/40",
    default: "bg-zinc-700/30 text-zinc-300 border-zinc-600/40",
  };
  return (
    <span
      className={`inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full border ${map[tone]}`}
    >
      {children}
    </span>
  );
}

export function ProgressBar({ pct }: { pct: number }) {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div className="w-full h-2 bg-panel2 rounded-full overflow-hidden border border-edge">
      <div
        className="h-full bg-gradient-to-r from-info to-accent transition-all"
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
}
