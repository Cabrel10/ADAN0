// API client + types for ADAN0 Terminal.

export interface ProcessInfo {
  running: boolean;
  pid: number | null;
  cpu_percent?: number;
  memory_percent?: number;
  elapsed_sec?: number;
}

export interface TrainingStatus {
  process: ProcessInfo;
  last_timestep: number | null;
  target_steps: number;
  progress_pct: number;
  steps_per_min: number | null;
  has_errors: boolean;
  error_count: number;
  last_error: string | null;
}

export interface TelemetryRow {
  timesteps: number;
  a0_mean: number | null;
  a0_std: number | null;
  a0_pct_buy: number | null;
  a0_pct_sell: number | null;
  a0_pct_hold_band: number | null;
  req_HOLD_pct: number | null;
  req_BUY_pct: number | null;
  req_SELL_pct: number | null;
  steps_flat_pct: number | null;
  steps_open_pct: number | null;
  illegal_ratio: number | null;
  policy_entropy: number | null;
  a0_histo: number[];
}

export interface CollapseVerdict {
  status: string;
  level: string;
  reasons: string[];
  a0_std_trend: number | null;
  latest: TelemetryRow | null;
  samples?: number;
}

export interface SystemStats {
  cpu_percent: number;
  cpu_count: number;
  mem_total_gb: number;
  mem_used_gb: number;
  mem_available_gb: number;
  mem_percent: number;
  swap_total_gb: number;
  swap_used_gb: number;
  swap_percent: number;
}

export interface Checkpoint {
  name: string;
  step: number | null;
  size_mb: number;
  mtime: number;
}

export interface SafeConfig {
  fees: { commission: number | null; round_trip_fees: number | null };
  reward_shaping: {
    invalid_trade_penalty_weight: number | null;
    sterile_action_geom_ratio: number | null;
    sterile_action_penalty_cap: number | null;
  };
  sandbox: { ent_coef: number | null };
  profile: string;
  asset: string;
  leverage: number;
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return (await r.json()) as T;
}

export const api = {
  health: () => get<{ status: string }>("/api/health"),
  status: () => get<TrainingStatus>("/api/training/status"),
  telemetry: (since = 0) =>
    get<{ rows: TelemetryRow[]; count: number }>(
      `/api/training/telemetry?since=${since}`
    ),
  collapse: () => get<CollapseVerdict>("/api/training/collapse"),
  log: (tail = 200) => get<{ lines: string[] }>(`/api/training/log?tail=${tail}`),
  checkpoints: () =>
    get<{ checkpoints: Checkpoint[]; count: number }>("/api/checkpoints"),
  config: () => get<SafeConfig>("/api/config"),
  system: () => get<SystemStats>("/api/system"),
};
