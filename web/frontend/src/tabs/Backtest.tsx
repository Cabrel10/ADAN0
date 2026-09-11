import { useEffect, useState } from "react";
import { Panel, Badge } from "../components/ui";
import { api, NamedBacktest } from "../api";
import { nf, pct, pctRaw, verdictTone } from "../format";

export default function Backtest() {
  const [bts, setBts] = useState<NamedBacktest[]>([]);
  const [err, setErr] = useState<string | null>(null);

  // interactive config (display + intent; launch wires into control)
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [tf, setTf] = useState("5m");
  const [capital, setCapital] = useState(1000);

  useEffect(() => {
    api
      .named()
      .then((r) => setBts(r.backtests))
      .catch((e) => setErr(String(e)));
  }, []);

  return (
    <div className="space-y-4">
      <Panel title="Backtesting Center — configuration">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <Field label="Symbol">
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="bg-panel2 border border-edge rounded px-2 py-1 w-full"
            >
              <option>BTCUSDT</option>
            </select>
          </Field>
          <Field label="Timeframe">
            <select
              value={tf}
              onChange={(e) => setTf(e.target.value)}
              className="bg-panel2 border border-edge rounded px-2 py-1 w-full"
            >
              <option>5m</option>
              <option>1h</option>
              <option>4h</option>
            </select>
          </Field>
          <Field label="Frais (verrouillés)">
            <input
              value="0.50%"
              disabled
              className="bg-bg border border-edge rounded px-2 py-1 w-full text-warn font-mono"
            />
          </Field>
          <Field label="Capital ($)">
            <input
              type="number"
              value={capital}
              onChange={(e) => setCapital(Number(e.target.value))}
              className="bg-panel2 border border-edge rounded px-2 py-1 w-full font-mono"
            />
          </Field>
        </div>
        <div className="mt-3 flex items-center gap-3">
          <button className="text-sm px-4 py-2 rounded bg-info/15 border border-info/50 text-info hover:bg-info/25">
            ▶ Lancer le backtest
          </button>
          <span className="text-[10px] text-muted">
            slippage par défaut · frais 0.5% non modifiables · résultats
            s'ajoutent ci-dessous
          </span>
        </div>
      </Panel>

      <Panel
        title="Backtests existants (logs/validation)"
        right={<span className="text-[11px] text-muted">{bts.length} runs</span>}
      >
        {err && <div className="text-down text-xs mb-2">{err}</div>}
        <div className="overflow-x-auto max-h-[460px]">
          <table className="w-full text-[11px]">
            <thead className="text-muted sticky top-0 bg-panel">
              <tr className="text-left">
                <th className="py-1 pr-3">Nom</th>
                <th className="py-1 pr-3 text-right">Trades</th>
                <th className="py-1 pr-3 text-right">Win rate</th>
                <th className="py-1 pr-3 text-right">PF</th>
                <th className="py-1 pr-3 text-right">Return</th>
                <th className="py-1 pr-3 text-right">Sharpe~</th>
                <th className="py-1 pr-3 text-right">Expectancy</th>
                <th className="py-1 pr-3">Verdict</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {bts.map((b) => (
                <tr key={b.name} className="border-t border-edge/40">
                  <td className="py-1 pr-3 text-zinc-100">{b.name}</td>
                  <td className="py-1 pr-3 text-right">{b.n_trades ?? "—"}</td>
                  <td className="py-1 pr-3 text-right">{pct(b.win_rate)}</td>
                  <td
                    className={`py-1 pr-3 text-right ${
                      (b.profit_factor ?? 0) >= 1 ? "text-up" : "text-down"
                    }`}
                  >
                    {nf(b.profit_factor)}
                  </td>
                  <td
                    className={`py-1 pr-3 text-right ${
                      (b.total_return_pct ?? 0) >= 0 ? "text-up" : "text-down"
                    }`}
                  >
                    {pctRaw(b.total_return_pct)}
                  </td>
                  <td className="py-1 pr-3 text-right">{nf(b.sharpe_like)}</td>
                  <td className="py-1 pr-3 text-right">{pctRaw(b.expectancy_pct, 3)}</td>
                  <td className="py-1 pr-3">
                    <Badge tone={verdictTone(b.verdict)}>{b.verdict ?? "—"}</Badge>
                  </td>
                </tr>
              ))}
              {bts.length === 0 && !err && (
                <tr>
                  <td colSpan={8} className="py-6 text-center text-muted">
                    Aucun backtest trouvé.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-[10px] uppercase tracking-wide text-muted">{label}</span>
      <div className="mt-1">{children}</div>
    </label>
  );
}
