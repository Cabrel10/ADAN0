import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Panel, Stat } from "../components/ui";
import { api, MetricsResponse } from "../api";
import { nf, pct, usd, goodHigh } from "../format";

interface EqPt {
  i: number;
  equity: number;
}

export default function Metrics() {
  const [m, setM] = useState<MetricsResponse | null>(null);
  const [eq, setEq] = useState<EqPt[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.metrics().then(setM).catch((e) => setErr(String(e)));
    api
      .equity()
      .then((r) => setEq(r.points.map((p) => ({ i: p.i, equity: p.equity }))))
      .catch(() => {});
  }, []);

  if (err) return <div className="text-down text-sm">{err}</div>;
  if (!m) return <div className="text-muted text-sm">Chargement des métriques…</div>;

  const met = m.metrics;
  const c = m.confusion;
  const peak = eq.length ? Math.max(...eq.map((p) => p.equity)) : 0;

  if (met.n_closed === 0) {
    return (
      <Panel title="Métriques">
        <div className="text-muted text-sm">
          {met.note ?? "Aucun trade clôturé trouvé."}
        </div>
      </Panel>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <Stat label="Trades clôturés" value={met.n_closed ?? "—"} />
        <Stat label="Win rate" value={pct(met.win_rate)} tone={goodHigh((met.win_rate ?? 0) - 0.5)} />
        <Stat label="Profit Factor" value={nf(met.profit_factor)} tone={goodHigh((met.profit_factor ?? 0) - 1)} />
        <Stat label="Expectancy" value={pct(met.expectancy, 3)} tone={goodHigh(met.expectancy)} />
        <Stat label="Total Return" value={pct(met.total_return)} tone={goodHigh(met.total_return)} />
        <Stat label="Max DD" value={pct(met.max_drawdown)} tone={goodHigh(met.max_drawdown)} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Panel title="Risk-adjusted">
          <table className="w-full text-xs">
            <tbody className="font-mono">
              <Row label="Sharpe" value={nf(met.sharpe, 3)} tone={(met.sharpe ?? 0) > 0} />
              <Row label="Sortino" value={nf(met.sortino, 3)} tone={(met.sortino ?? 0) > 0} />
              <Row label="Calmar" value={nf(met.calmar, 3)} tone={(met.calmar ?? 0) > 0} />
              <Row label="Mean return" value={pct(met.mean_return, 3)} tone={(met.mean_return ?? 0) > 0} />
              <Row label="Std return" value={pct(met.std_return, 3)} />
            </tbody>
          </table>
        </Panel>

        <Panel title="Tail risk">
          <table className="w-full text-xs">
            <tbody className="font-mono">
              <Row label="VaR 95%" value={pct(met.var95, 3)} tone={(met.var95 ?? 0) > 0} />
              <Row label="CVaR 95%" value={pct(met.cvar95, 3)} tone={(met.cvar95 ?? 0) > 0} />
              <Row label="Best trade" value={pct(met.best, 3)} tone />
              <Row label="Worst trade" value={pct(met.worst, 3)} tone={false} />
              <Row label="Max consec. losses" value={String(met.max_consecutive_losses ?? "—")} tone={false} />
            </tbody>
          </table>
        </Panel>

        <Panel title="Confusion (entrées / sorties)">
          <div className="grid grid-cols-2 gap-2 text-center">
            <Cell label="BUY open" value={c.buy_open} color="#00FF88" />
            <Cell label="SELL open" value={c.sell_open} color="#FF4D4D" />
            <Cell label="CLOSE win" value={c.close_win} color="#00FF88" />
            <Cell label="CLOSE loss" value={c.close_loss} color="#FF4D4D" />
          </div>
          <div className="text-[10px] text-muted mt-3">
            {met.n_wins ?? 0} gains / {met.n_losses ?? 0} pertes
          </div>
        </Panel>
      </div>

      <Panel
        title="Equity curve (reconstruite depuis les trades)"
        right={<span className="text-[11px] text-muted">peak ${nf(peak)}</span>}
      >
        <div style={{ height: 300 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={eq} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid stroke="rgba(36,51,82,0.4)" />
              <XAxis dataKey="i" stroke="#64748B" fontSize={10} />
              <YAxis stroke="#64748B" fontSize={10} domain={["auto", "auto"]} />
              <Tooltip
                contentStyle={{
                  background: "#111B2E",
                  border: "1px solid #243352",
                  fontSize: 11,
                }}
                formatter={(v: number) => usd(v)}
              />
              <ReferenceLine y={1000} stroke="#64748B" strokeDasharray="3 3" />
              <Line
                type="monotone"
                dataKey="equity"
                stroke="#3B82F6"
                strokeWidth={1.6}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="text-[10px] text-muted mt-2">
          source: <code>{m.file}</code> · capital de départ $1000
        </div>
      </Panel>
    </div>
  );
}

function Row({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: boolean;
}) {
  const cls = tone === undefined ? "text-zinc-100" : tone ? "text-up" : "text-down";
  return (
    <tr className="border-t border-edge/40">
      <td className="py-1.5 text-muted">{label}</td>
      <td className={`py-1.5 text-right ${cls}`}>{value}</td>
    </tr>
  );
}

function Cell({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="bg-panel2 border border-edge rounded-md py-3">
      <div className="text-xl font-semibold" style={{ color }}>
        {value}
      </div>
      <div className="text-[10px] text-muted">{label}</div>
    </div>
  );
}
