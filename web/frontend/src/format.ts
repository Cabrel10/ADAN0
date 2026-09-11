// Formatting helpers for Mission Control.

export const nf = (v: number | null | undefined, d = 2): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(d);

export const pct = (v: number | null | undefined, d = 2): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`;

// value already in percent units (e.g. 12.3 means 12.3%)
export const pctRaw = (v: number | null | undefined, d = 2): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${v.toFixed(d)}%`;

export const usd = (v: number | null | undefined, d = 2): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `$${v.toFixed(d)}`;

export const intf = (v: number | null | undefined): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : Math.round(v).toLocaleString();

// tone for a metric where higher is better
export const goodHigh = (v: number | null | undefined, mid = 0): "up" | "down" | "default" => {
  if (v === null || v === undefined || Number.isNaN(v)) return "default";
  return v > mid ? "up" : "down";
};

export const sideColor = (side?: string, reason?: string): string => {
  const r = (reason || "").toUpperCase();
  const s = (side || "").toUpperCase();
  if (r.includes("CLOSE") || r === "TP" || r === "SL" || r.includes("EXIT")) return "#FFC857";
  if (s === "BUY" || s === "LONG") return "#00FF88";
  if (s === "SELL" || s === "SHORT") return "#FF4D4D";
  return "#3B82F6";
};

export const verdictTone = (
  v: string | null | undefined
): "ok" | "warn" | "crit" | "info" | "default" => {
  if (!v) return "default";
  const u = v.toUpperCase();
  if (u.includes("GOOD") || u.includes("PASS") || u.includes("PROFITABLE") || u.includes("OK"))
    return "ok";
  if (u.includes("WARN") || u.includes("MARGINAL") || u.includes("RISK")) return "warn";
  if (u.includes("FAIL") || u.includes("BAD") || u.includes("LOSS") || u.includes("NO_TRADES"))
    return "crit";
  return "info";
};
