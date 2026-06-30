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

// ---- Analytics types ----
export interface RegistryModel {
  checkpoint: number;
  n_trades: number | null;
  win_rate: number | null;
  profit_factor: number | null;
  expectancy_pct: number | null;
  total_return_pct: number | null;
  sharpe_like: number | null;
  best_trade_pct: number | null;
  worst_trade_pct: number | null;
  max_consecutive_losses: number | null;
  verdict: string;
  source: string;
}

export interface TradeRow {
  timestamp?: string;
  side?: string;
  symbol?: string;
  price?: number;
  size_usd?: number;
  size_asset?: number;
  sl_pct?: number;
  tp_pct?: number;
  fee_usd?: number;
  pnl_usd?: number;
  reason?: string;
  source?: string;
  order_id?: string;
}

export interface Metrics {
  n_closed: number;
  win_rate?: number;
  profit_factor?: number;
  expectancy?: number;
  mean_return?: number;
  std_return?: number;
  sharpe?: number;
  sortino?: number;
  calmar?: number;
  max_drawdown?: number;
  total_return?: number;
  best?: number;
  worst?: number;
  var95?: number;
  cvar95?: number;
  max_consecutive_losses?: number;
  n_wins?: number;
  n_losses?: number;
  note?: string;
}

export interface Confusion {
  buy_open: number;
  sell_open: number;
  close_win: number;
  close_loss: number;
}

export interface MetricsResponse {
  file: string | null;
  metrics: Metrics;
  confusion: Confusion;
}

export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface NamedBacktest {
  name: string;
  n_trades: number | null;
  win_rate: number | null;
  profit_factor: number | null;
  total_return_pct: number | null;
  sharpe_like: number | null;
  expectancy_pct: number | null;
  verdict: string | null;
}

// ---- Control types ----
export interface ControlStatus {
  status: {
    running: boolean;
    pid: number | null;
    cmdline?: string;
    cpu_percent?: number;
    memory_percent?: number;
  };
  workers: string[];
  editable_hyperparams: string[];
  forbidden: string[];
}

export interface LaunchRequest {
  steps: number;
  worker: string;
  hyperparams: Record<string, number>;
  diag: boolean;
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return (await r.json()) as T;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      const j = await r.json();
      detail = j.detail || JSON.stringify(j);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
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
  // analytics
  registry: () => get<{ models: RegistryModel[] }>("/api/analytics/registry"),
  named: () => get<{ backtests: NamedBacktest[] }>("/api/analytics/named"),
  trades: (limit = 2000) =>
    get<{ file: string | null; count: number; trades: TradeRow[] }>(
      `/api/analytics/trades?limit=${limit}`
    ),
  metrics: () => get<MetricsResponse>("/api/analytics/metrics"),
  candles: (timeframe = "5m", limit = 500) =>
    get<{ timeframe: string; count: number; candles: Candle[] }>(
      `/api/analytics/candles?timeframe=${timeframe}&limit=${limit}`
    ),
  equity: () =>
    get<{ points: { i: number; equity: number }[] }>("/api/analytics/equity"),
  // control
  controlStatus: () => get<ControlStatus>("/api/control/status"),
  launch: (req: LaunchRequest) =>
    post<{ ok: boolean; pid?: number; message?: string }>(
      "/api/control/launch",
      req
    ),
  stop: () => post<{ ok: boolean; message?: string }>("/api/control/stop", {}),
};
