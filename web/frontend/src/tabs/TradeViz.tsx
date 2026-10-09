import { useEffect, useState } from "react";
import { Panel, Badge } from "../components/ui";
import CandleChart from "../components/CandleChart";
import { api, Candle, TradeRow } from "../api";
import { usd, pct, sideColor } from "../format";

const TIMEFRAMES = ["5m", "1h", "4h"] as const;

export default function TradeViz() {
  const [tf, setTf] = useState<(typeof TIMEFRAMES)[number]>("5m");
  const [candles, setCandles] = useState<Candle[]>([]);
  const [trades, setTrades] = useState<TradeRow[]>([]);
  const [showTrades, setShowTrades] = useState(true);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [file, setFile] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setErr(null);
    api
      .candles(tf, 800)
      .then((r) => {
        if (alive) setCandles(r.candles);
      })
      .catch((e) => alive && setErr(String(e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [tf]);

  useEffect(() => {
    let alive = true;
    api
      .trades(2000)
      .then((r) => {
        if (alive) {
          setTrades(r.trades);
          setFile(r.file);
        }
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  const markerTrades = showTrades ? trades : [];

  return (
    <div className="space-y-4">
      <Panel
        title={`Visualisation des positions · BTCUSDT ${tf}`}
        right={
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowTrades((v) => !v)}
              className={`text-[11px] px-2 py-0.5 rounded border ${
                showTrades
                  ? "border-up/50 text-up bg-up/10"
                  : "border-edge text-muted"
              }`}
            >
              {showTrades ? "● markers ON" : "○ markers OFF"}
            </button>
            <div className="flex gap-1">
              {TIMEFRAMES.map((t) => (
                <button
                  key={t}
                  onClick={() => setTf(t)}
                  className={`text-[11px] px-2 py-0.5 rounded border ${
                    tf === t
                      ? "border-accent text-zinc-100 bg-accent/15"
                      : "border-edge text-muted hover:text-zinc-300"
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>
        }
      >
        <div className="flex items-center gap-4 mb-2 text-[11px]">
          <span className="flex items-center gap-1">
            <span style={{ color: "#00FF88" }}>▲</span> BUY
          </span>
          <span className="flex items-center gap-1">
            <span style={{ color: "#FF4D4D" }}>▼</span> SELL
          </span>
          <span className="flex items-center gap-1">
            <span style={{ color: "#FFC857" }}>●</span> CLOSE
          </span>
          <span className="text-muted">
            {candles.length} bougies · {trades.length} trades
          </span>
        </div>
        {err && <div className="text-down text-xs mb-2">{err}</div>}
        {loading ? (
          <div className="h-[420px] flex items-center justify-center text-muted text-xs">
            Chargement des bougies…
          </div>
        ) : (
          <CandleChart candles={candles} trades={markerTrades} height={460} />
        )}
        {file && (
          <div className="text-[10px] text-muted mt-2">
            trades source: <code>{file}</code> — les timestamps des trades sont
            calés sur la bougie la plus proche.
          </div>
        )}
      </Panel>

      <Panel title="Journal des positions (trade par trade)">
        <div className="overflow-x-auto max-h-[360px]">
          <table className="w-full text-[11px]">
            <thead className="text-muted sticky top-0 bg-panel">
              <tr className="text-left">
                <th className="py-1 pr-3">#</th>
                <th className="py-1 pr-3">Timestamp</th>
                <th className="py-1 pr-3">Side</th>
                <th className="py-1 pr-3">Reason</th>
                <th className="py-1 pr-3 text-right">Price</th>
                <th className="py-1 pr-3 text-right">Size $</th>
                <th className="py-1 pr-3 text-right">SL%</th>
                <th className="py-1 pr-3 text-right">TP%</th>
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
                    <td
                      className="py-1 pr-3 font-semibold"
                      style={{ color: sideColor(t.side, t.reason) }}
                    >
                      {t.side ?? "—"}
                    </td>
                    <td className="py-1 pr-3 text-muted">{t.reason ?? "—"}</td>
                    <td className="py-1 pr-3 text-right">{usd(t.price)}</td>
                    <td className="py-1 pr-3 text-right">{usd(t.size_usd)}</td>
                    <td className="py-1 pr-3 text-right text-muted">
                      {t.sl_pct != null ? pct(t.sl_pct) : "—"}
                    </td>
                    <td className="py-1 pr-3 text-right text-muted">
                      {t.tp_pct != null ? pct(t.tp_pct) : "—"}
                    </td>
                    <td className="py-1 pr-3 text-right text-muted">
                      {usd(t.fee_usd)}
                    </td>
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
              {trades.length === 0 && (
                <tr>
                  <td colSpan={10} className="py-6 text-center text-muted">
                    Aucun trade chargé.
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
