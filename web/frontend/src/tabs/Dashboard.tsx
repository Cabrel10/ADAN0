import { Panel, Stat, Badge, ProgressBar } from "../components/ui";
import { usePoll } from "../usePoll";
import { api } from "../api";

function fmtElapsed(s?: number) {
  if (!s) return "—";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

export default function Dashboard() {
  const { data: status } = usePoll(api.status, 3000);
  const { data: sys } = usePoll(api.system, 4000);
  const { data: collapse } = usePoll(api.collapse, 5000);
  const { data: cfg } = usePoll(api.config, 30000);
  const { data: cks } = usePoll(api.checkpoints, 15000);

  const running = status?.process.running;
  const collapseTone =
    collapse?.level === "critical"
      ? "crit"
      : collapse?.level === "warning"
      ? "warn"
      : collapse?.level === "ok"
      ? "ok"
      : "default";

  return (
    <div className="space-y-4">
      <Panel
        title="Run actif — DIAGNOSTIC-V4 500k"
        right={
          <Badge tone={running ? "ok" : "crit"}>
            <span className={running ? "pulse-dot" : ""}>●</span>
            {running ? "TRAINING" : "STOPPED"}
          </Badge>
        }
      >
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Stat
            label="Timestep"
            value={(status?.last_timestep ?? 0).toLocaleString()}
            sub={`/ ${(status?.target_steps ?? 500000).toLocaleString()}`}
            tone="info"
          />
          <Stat
            label="Progression"
            value={`${status?.progress_pct ?? 0}%`}
            sub={<ProgressBar pct={status?.progress_pct ?? 0} />}
          />
          <Stat
            label="Vitesse"
            value={status?.steps_per_min ? `${status.steps_per_min}` : "—"}
            sub="steps / min"
          />
          <Stat
            label="Écoulé"
            value={fmtElapsed(status?.process.elapsed_sec)}
            sub={status?.process.pid ? `PID ${status.process.pid}` : "—"}
          />
        </div>
      </Panel>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Panel
          title="Santé collapse"
          right={<Badge tone={collapseTone}>{collapse?.status ?? "…"}</Badge>}
        >
          <ul className="text-xs space-y-1 text-zinc-300">
            {(collapse?.reasons ?? ["…"]).map((r, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-muted">›</span>
                {r}
              </li>
            ))}
          </ul>
          {collapse?.latest && (
            <div className="mt-3 grid grid-cols-3 gap-2 text-center">
              <MiniStat label="a0_std" v={collapse.latest.a0_std} />
              <MiniStat label="HOLD%" v={collapse.latest.req_HOLD_pct} />
              <MiniStat label="illegal" v={collapse.latest.illegal_ratio} />
            </div>
          )}
        </Panel>

        <Panel title="Erreurs runtime">
          <div className="flex items-center gap-3">
            <Badge tone={status?.has_errors ? "crit" : "ok"}>
              {status?.has_errors ? "ERREURS" : "AUCUNE"}
            </Badge>
            <span className="text-sm text-muted">
              {status?.error_count ?? 0} occurrence(s)
            </span>
          </div>
          {status?.last_error && (
            <pre className="mt-2 text-[10px] text-down whitespace-pre-wrap break-all">
              {status.last_error}
            </pre>
          )}
          {!status?.has_errors && (
            <p className="text-xs text-muted mt-2">
              Le fix NameError '_base' tient — aucun crash par step.
            </p>
          )}
        </Panel>

        <Panel title="Frais (verrouillés)">
          <div className="grid grid-cols-2 gap-2">
            <MiniStat label="commission" v={cfg?.fees.commission} fixed={4} />
            <MiniStat label="round_trip" v={cfg?.fees.round_trip_fees} fixed={4} />
          </div>
          <div className="grid grid-cols-2 gap-2 mt-2">
            <MiniStat
              label="inv_pen_w"
              v={cfg?.reward_shaping.invalid_trade_penalty_weight}
              fixed={3}
            />
            <MiniStat label="ent_coef" v={cfg?.sandbox.ent_coef} fixed={2} />
          </div>
          <p className="text-[10px] text-muted mt-2">
            {cfg?.profile} · {cfg?.asset} · lev {cfg?.leverage}
          </p>
        </Panel>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Panel title="Mémoire VPS" className="md:col-span-2">
          <div className="grid grid-cols-3 gap-3">
            <Stat
              label="RAM dispo"
              value={`${sys?.mem_available_gb ?? "—"} Gi`}
              sub={`${sys?.mem_percent ?? 0}% utilisée`}
              tone={
                (sys?.mem_available_gb ?? 9) < 0.5 ? "down" : "default"
              }
            />
            <Stat
              label="Swap"
              value={`${sys?.swap_used_gb ?? "—"} / ${sys?.swap_total_gb ?? "—"}`}
              sub={`${sys?.swap_percent ?? 0}%`}
              tone={(sys?.swap_percent ?? 0) > 90 ? "warn" : "default"}
            />
            <Stat
              label="CPU"
              value={`${sys?.cpu_percent ?? "—"}%`}
              sub={`${sys?.cpu_count ?? "—"} vCPU`}
            />
          </div>
        </Panel>
        <Panel title="Checkpoints" right={<Badge tone="info">{cks?.count ?? 0}</Badge>}>
          <ul className="text-xs space-y-1 max-h-32 overflow-auto">
            {(cks?.checkpoints ?? []).slice(0, 6).map((c) => (
              <li key={c.name} className="flex justify-between gap-2">
                <span className="text-zinc-300 truncate">
                  {c.step != null ? `@${c.step.toLocaleString()}` : c.name}
                </span>
                <span className="text-muted">{c.size_mb}MB</span>
              </li>
            ))}
            {!cks?.checkpoints.length && (
              <li className="text-muted">aucun checkpoint</li>
            )}
          </ul>
        </Panel>
      </div>
    </div>
  );
}

function MiniStat({
  label,
  v,
  fixed = 3,
}: {
  label: string;
  v: number | null | undefined;
  fixed?: number;
}) {
  return (
    <div className="bg-panel2 border border-edge rounded px-2 py-1 text-center">
      <div className="text-[9px] uppercase text-muted">{label}</div>
      <div className="text-sm text-zinc-100">
        {v == null ? "—" : v.toFixed(fixed)}
      </div>
    </div>
  );
}
