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
import { Panel, Stat } from "../components/ui";
import { api, TradeRow } from "../api";
import { usd, pct, sideColor, goodHigh } from "../format";

interface EqPt {
  i: number;
  equity: number;
}

export default function Paper() {
  const [trades, setTrades] = useState<TradeRow[]>([]);
  const [eq, setEq] = useState<EqPt[]>([]);
  const [file, setFile] = useState<string | null>(null);

  useEffect(() => {
    api.trades(2000).then((r) => {
      setTrades(r.trades);
      setFile(r.file);
    });
    api.equity().then((r) => setEq(r.points.map((p) => ({ i: p.i, equity: p.equity }))));
  }, []);

  const closed = trades.filter((t) => (t.pnl_usd ?? 0) !== 0);
  const totalPnl = closed.reduce((s, t) => s + (t.pnl_usd ?? 0), 0);
  const totalFees = trades.reduce((s, t) => s + (t.fee_usd ?? 0), 0);
  const wins = closed.filter((t) => (t.pnl_usd ?? 0) > 0).length;
  const wr = closed.length ? wins / closed.length : 0;
  const lastEq = eq.length ? eq[eq.length - 1].equity : 1000;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-5 gap-3">
        <Stat label="Equity" value={usd(lastEq)} tone={goodHigh(lastEq - 1000)} />
        <Stat label="PnL net" value={usd(totalPnl)} tone={goodHigh(totalPnl)} />
        <Stat label="Frais payés" value={usd(totalFees)} tone="warn" />
        <Stat label="Win rate" value={pct(wr)} tone={goodHigh(wr - 0.5)} />
        <Stat label="Trades clôturés" value={closed.length} />
      </div>

      <Panel title="Equity (paper)" right={<span className="text-[11px] text-muted">capital départ $1000</span>}>
        <div style={{ height: 260 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={eq} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
              <defs>
                <linearGradient id="eqfill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3B82F6" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#3B82F6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(36,51,82,0.4)" />
              <XAxis dataKey="i" stroke="#64748B" fontSize={10} />
              <YAxis stroke="#64748B" fontSize={10} domain={["auto", "auto"]} />
              <Tooltip
                contentStyle={{ background: "#111B2E", border: "1px solid #243352", fontSize: 11 }}
                formatter={(v: number) => usd(v)}
              />
              <ReferenceLine y={1000} stroke="#64748B" strokeDasharray="3 3" />
              <Area type="monotone" dataKey="equity" stroke="#3B82F6" strokeWidth={1.6} fill="url(#eqfill)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <Panel
        title="Historique des ordres (paper)"
        right={file ? <span className="text-[10px] text-muted">{file}</span> : null}
      >
        <div className="overflow-x-auto max-h-[420px]">
          <table className="w-full text-[11px]">
            <thead className="text-muted sticky top-0 bg-panel">
              <tr className="text-left">
                <th className="py-1 pr-3">#</th>
                <th className="py-1 pr-3">Timestamp</th>
                <th className="py-1 pr-3">Side</th>
                <th className="py-1 pr-3">Reason</th>
                <th className="py-1 pr-3 text-right">Price</th>
                <th className="py-1 pr-3 text-right">Size $</th>
                <th className="py-1 pr-3 text-right">Fee</th>
                <th className="py-1 pr-3 text-right">PnL $</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {trades.map((t, i) => {
                const pnl = t.pnl_usd ?? 0;
                return (
                  <tr key={i} className="border-t border-edge/40">
                    <td className="py-1 pr-3 text-muted">{i + 1}</td>
                    <td className="py-1 pr-3 text-muted">{t.timestamp ?? "—"}</td>
                    <td className="py-1 pr-3 font-semibold" style={{ color: sideColor(t.side, t.reason) }}>
                      {t.side ?? "—"}
                    </td>
                    <td className="py-1 pr-3 text-muted">{t.reason ?? "—"}</td>
                    <td className="py-1 pr-3 text-right">{usd(t.price)}</td>
                    <td className="py-1 pr-3 text-right">{usd(t.size_usd)}</td>
                    <td className="py-1 pr-3 text-right text-muted">{usd(t.fee_usd)}</td>
                    <td
                      className={`py-1 pr-3 text-right font-semibold ${
                        pnl > 0 ? "text-up" : pnl < 0 ? "text-down" : "text-muted"
                      }`}
                    >
                      {usd(t.pnl_usd)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
