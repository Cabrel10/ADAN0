import { useEffect, useState } from "react";
import { Panel, Badge } from "../components/ui";
import { api, RegistryModel } from "../api";
import { nf, pct, pctRaw, intf, verdictTone } from "../format";

export default function Models() {
  const [models, setModels] = useState<RegistryModel[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [sel, setSel] = useState<number[]>([]);

  useEffect(() => {
    api
      .registry()
      .then((r) => setModels(r.models))
      .catch((e) => setErr(String(e)));
  }, []);

  const toggle = (cp: number) =>
    setSel((s) =>
      s.includes(cp) ? s.filter((x) => x !== cp) : s.length < 3 ? [...s, cp] : s
    );

  const compare = models.filter((m) => sel.includes(m.checkpoint));

  if (err) return <div className="text-down text-sm">{err}</div>;

  return (
    <div className="space-y-4">
      <Panel
        title="Model Registry — backtest par checkpoint"
        right={
          <span className="text-[11px] text-muted">
            {models.length} modèles · sélectionnez jusqu'à 3 pour comparer
          </span>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-[11px]">
            <thead className="text-muted">
              <tr className="text-left">
                <th className="py-1 pr-2"></th>
                <th className="py-1 pr-3">Checkpoint</th>
                <th className="py-1 pr-3 text-right">Trades</th>
                <th className="py-1 pr-3 text-right">Win rate</th>
                <th className="py-1 pr-3 text-right">PF</th>
                <th className="py-1 pr-3 text-right">Expectancy</th>
                <th className="py-1 pr-3 text-right">Return</th>
                <th className="py-1 pr-3 text-right">Sharpe~</th>
                <th className="py-1 pr-3 text-right">Best</th>
                <th className="py-1 pr-3 text-right">Worst</th>
                <th className="py-1 pr-3">Verdict</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {models.map((m) => (
                <tr
                  key={m.source}
                  className={`border-t border-edge/40 ${
                    sel.includes(m.checkpoint) ? "bg-accent/10" : ""
                  }`}
                >
                  <td className="py-1 pr-2">
                    <input
                      type="checkbox"
                      checked={sel.includes(m.checkpoint)}
                      onChange={() => toggle(m.checkpoint)}
                      className="accent-violet-500"
                    />
                  </td>
                  <td className="py-1 pr-3 text-zinc-100">
                    PPO_{intf(m.checkpoint)}
                  </td>
                  <td className="py-1 pr-3 text-right">{m.n_trades ?? "—"}</td>
                  <td className="py-1 pr-3 text-right">{pct(m.win_rate)}</td>
                  <td
                    className={`py-1 pr-3 text-right ${
                      (m.profit_factor ?? 0) >= 1 ? "text-up" : "text-down"
                    }`}
                  >
                    {nf(m.profit_factor)}
                  </td>
                  <td className="py-1 pr-3 text-right">{pctRaw(m.expectancy_pct, 3)}</td>
                  <td
                    className={`py-1 pr-3 text-right ${
                      (m.total_return_pct ?? 0) >= 0 ? "text-up" : "text-down"
                    }`}
                  >
                    {pctRaw(m.total_return_pct)}
                  </td>
                  <td className="py-1 pr-3 text-right">{nf(m.sharpe_like)}</td>
                  <td className="py-1 pr-3 text-right text-up">{pctRaw(m.best_trade_pct)}</td>
                  <td className="py-1 pr-3 text-right text-down">{pctRaw(m.worst_trade_pct)}</td>
                  <td className="py-1 pr-3">
                    <Badge tone={verdictTone(m.verdict)}>{m.verdict ?? "—"}</Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {compare.length >= 2 && (
        <Panel title={`Comparaison (${compare.length} modèles)`}>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-muted text-left">
                  <th className="py-1 pr-4">Métrique</th>
                  {compare.map((m) => (
                    <th key={m.source} className="py-1 pr-4 text-right">
                      PPO_{intf(m.checkpoint)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="font-mono">
                <CmpRow label="Trades" models={compare} get={(m) => m.n_trades} fmt={(v) => String(v ?? "—")} />
                <CmpRow label="Win rate" models={compare} get={(m) => m.win_rate} fmt={(v) => pct(v)} hi />
                <CmpRow label="Profit Factor" models={compare} get={(m) => m.profit_factor} fmt={(v) => nf(v)} hi />
                <CmpRow label="Expectancy %" models={compare} get={(m) => m.expectancy_pct} fmt={(v) => pctRaw(v, 3)} hi />
                <CmpRow label="Return %" models={compare} get={(m) => m.total_return_pct} fmt={(v) => pctRaw(v)} hi />
                <CmpRow label="Sharpe~" models={compare} get={(m) => m.sharpe_like} fmt={(v) => nf(v)} hi />
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}

function CmpRow({
  label,
  models,
  get,
  fmt,
  hi = false,
}: {
  label: string;
  models: RegistryModel[];
  get: (m: RegistryModel) => number | null;
  fmt: (v: number | null) => string;
  hi?: boolean;
}) {
  const vals = models.map(get).filter((v): v is number => v != null);
  const best = hi && vals.length ? Math.max(...vals) : null;
  return (
    <tr className="border-t border-edge/40">
      <td className="py-1.5 text-muted">{label}</td>
      {models.map((m) => {
        const v = get(m);
        const isBest = best != null && v === best;
        return (
          <td
            key={m.source}
            className={`py-1.5 pr-4 text-right ${isBest ? "text-up font-semibold" : "text-zinc-100"}`}
          >
            {fmt(v)}
            {isBest && " ★"}
          </td>
        );
      })}
    </tr>
  );
}
