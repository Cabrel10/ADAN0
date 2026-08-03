import { useEffect, useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Panel, Badge } from "../components/ui";
import {
  api,
  ValidationResult,
  MarkerValidation,
  EqDDPoint,
  PerfBlock,
  RLBlock,
  Provenance,
} from "../api";
import { nf, usd } from "../format";

function isProv(v: unknown): v is Provenance {
  return typeof v === "object" && v !== null && "value" in (v as object);
}

export default function Reliability() {
  const [val, setVal] = useState<ValidationResult | null>(null);
  const [mk, setMk] = useState<MarkerValidation | null>(null);
  const [eqdd, setEqdd] = useState<EqDDPoint[]>([]);
  const [perf, setPerf] = useState<PerfBlock | null>(null);
  const [rl, setRl] = useState<RLBlock | null>(null);

  useEffect(() => {
    api.metricsValidate().then(setVal).catch(() => {});
    api.metricsMarkers("5m").then(setMk).catch(() => {});
    api.metricsEquityDrawdown().then((r) => setEqdd(r.points)).catch(() => {});
    api.metricsPerformance().then(setPerf).catch(() => {});
    api.metricsRL().then(setRl).catch(() => {});
  }, []);

  return (
    <div className="space-y-4">
      <Panel
        title="Metrics Validator — Dashboard vs Recompute indépendant"
        right={
          val ? (
            <Badge tone={val.all_match ? "ok" : "crit"}>
              {val.all_match ? "✓ ALL MATCH" : "✗ MISMATCH"}
            </Badge>
          ) : null
        }
      >
        <table className="w-full text-xs">
          <thead className="text-muted text-left">
            <tr>
              <th className="py-1 pr-4">Metric</th>
              <th className="py-1 pr-4 text-right">Dashboard</th>
              <th className="py-1 pr-4 text-right">Recomputed</th>
              <th className="py-1 pr-4">Match</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {val &&
              Object.entries(val.checks).map(([k, c]) => (
                <tr key={k} className="border-t border-edge/40">
                  <td className="py-1.5 pr-4 text-muted">{k}</td>
                  <td className="py-1.5 pr-4 text-right text-zinc-100">
                    {c.dashboard ?? "—"}
                  </td>
                  <td className="py-1.5 pr-4 text-right text-zinc-100">
                    {c.recomputed ?? "—"}
                  </td>
                  <td className="py-1.5 pr-4">
                    <span className={c.match ? "text-up" : "text-down"}>
                      {c.match ? "✅" : "❌ Mismatch"}
                    </span>
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
        {val && (
          <div className="text-[10px] text-muted mt-2">
            source: <code>{val.source}</code> · validé {val.computed_at}
          </div>
        )}
      </Panel>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Panel
          title="Validation des marqueurs de trade"
          right={
            mk ? (
              <Badge tone={mk.invalid === 0 ? "ok" : "warn"}>
                {mk.valid}/{mk.checked} valides
              </Badge>
            ) : null
          }
        >
          {mk && (
            <div className="text-xs text-zinc-300 space-y-1">
              <div>
                Plage marché:{" "}
                <span className="font-mono text-info">
                  ${mk.market_range?.low} – ${mk.market_range?.high}
                </span>
              </div>
              <div>
                Chaque prix de trade est vérifié dans la plage réelle des bougies.
                {mk.invalid === 0 ? (
                  <span className="text-up"> Aucun marqueur corrompu.</span>
                ) : (
                  <span className="text-down"> {mk.invalid} hors plage.</span>
                )}
              </div>
              <div className="max-h-40 overflow-auto mt-2 font-mono text-[10px]">
                {mk.markers.slice(0, 30).map((m) => (
                  <div key={m.idx} className="flex justify-between border-b border-edge/30 py-0.5">
                    <span className="text-muted">
                      #{m.idx} {m.side} {m.reason}
                    </span>
                    <span>
                      {usd(m.price)}{" "}
                      <span className={m.in_market_range ? "text-up" : "text-down"}>
                        {m.status}
                      </span>
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Panel>

        <Panel title="RL metrics (avec provenance)">
          {rl && (
            <table className="w-full text-xs font-mono">
              <tbody>
                {Object.entries(rl)
                  .filter(([, v]) => isProv(v))
                  .map(([k, v]) => {
                    const p = v as Provenance;
                    return (
                      <tr key={k} className="border-t border-edge/40">
                        <td className="py-1 text-muted">{k}</td>
                        <td className="py-1 text-right text-zinc-100">
                          {p.value ?? "—"}
                        </td>
                        <td className="py-1 text-right text-[9px] text-muted pl-3">
                          @{p.window}
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          )}
          {rl && (
            <div className="text-[10px] text-muted mt-2">
              source: <code>{(rl as Record<string, string>).source}</code>
            </div>
          )}
        </Panel>
      </div>

      <Panel title="Equity & Drawdown (reconstruits depuis les trades bruts)">
        <div style={{ height: 220 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={eqdd} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="rgba(36,51,82,0.4)" />
              <XAxis dataKey="i" stroke="#64748B" fontSize={10} />
              <YAxis stroke="#64748B" fontSize={10} domain={["auto", "auto"]} />
              <Tooltip
                contentStyle={{ background: "#111B2E", border: "1px solid #243352", fontSize: 11 }}
                formatter={(v: number) => usd(v)}
              />
              <ReferenceLine y={1000} stroke="#64748B" strokeDasharray="3 3" />
              <Area type="monotone" dataKey="equity" stroke="#3B82F6" fill="rgba(59,130,246,0.15)" strokeWidth={1.5} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div style={{ height: 140 }} className="mt-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={eqdd} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="rgba(36,51,82,0.4)" />
              <XAxis dataKey="i" stroke="#64748B" fontSize={10} />
              <YAxis stroke="#64748B" fontSize={10} />
              <Tooltip
                contentStyle={{ background: "#111B2E", border: "1px solid #243352", fontSize: 11 }}
                formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
              />
              <Area type="monotone" dataKey="drawdown" stroke="#FF4D4D" fill="rgba(255,77,77,0.15)" strokeWidth={1.5} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="text-[10px] text-muted mt-1">
          Equity (bleu) · Drawdown (rouge). Tout est RECALCULÉ depuis le CSV de
          trades — jamais lu depuis une variable mémoire.
        </div>
      </Panel>

      <Panel title="Architecture de fiabilité">
        <div className="flex items-center gap-2 text-[11px] flex-wrap">
          {["Raw data (CSV/parquet)", "Metrics Engine", "Validator", "Cache (TTL)", "Dashboard / Agent"].map(
            (s, i, arr) => (
              <span key={s} className="flex items-center gap-2">
                <span className="px-2 py-1 rounded border border-edge bg-panel2 text-zinc-200">
                  {s}
                </span>
                {i < arr.length - 1 && <span className="text-accent">→</span>}
              </span>
            )
          )}
        </div>
        <div className="text-[10px] text-muted mt-2">
          Le dashboard ne calcule jamais directement les métriques : il lit des
          résultats déjà validés et horodatés.
        </div>
      </Panel>
    </div>
  );
}
