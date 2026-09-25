import { useEffect, useRef } from "react";
import {
  createChart,
  ColorType,
  CrosshairMode,
  IChartApi,
  ISeriesApi,
  Time,
} from "lightweight-charts";
import type { Candle, TradeRow } from "../api";

export interface TradeMarker {
  time: number; // unix seconds
  side?: string;
  reason?: string;
  price?: number;
}

function markerFor(m: TradeMarker) {
  const r = (m.reason || "").toUpperCase();
  const s = (m.side || "").toUpperCase();
  const isClose =
    r.includes("CLOSE") || r === "TP" || r === "SL" || r.includes("EXIT");
  if (isClose) {
    return {
      time: m.time as Time,
      position: "aboveBar" as const,
      color: "#FFC857",
      shape: "circle" as const,
      text: "CLOSE",
    };
  }
  if (s === "BUY" || s === "LONG") {
    return {
      time: m.time as Time,
      position: "belowBar" as const,
      color: "#00FF88",
      shape: "arrowUp" as const,
      text: "BUY",
    };
  }
  return {
    time: m.time as Time,
    position: "aboveBar" as const,
    color: "#FF4D4D",
    shape: "arrowDown" as const,
    text: "SELL",
  };
}

export default function CandleChart({
  candles,
  trades = [],
  height = 420,
}: {
  candles: Candle[];
  trades?: TradeRow[];
  height?: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volRef = useRef<ISeriesApi<"Histogram"> | null>(null);

  // create chart once
  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      height,
      layout: {
        background: { type: ColorType.Solid, color: "#0B1220" },
        textColor: "#CBD5E1",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(36,51,82,0.4)" },
        horzLines: { color: "rgba(36,51,82,0.4)" },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#243352" },
      timeScale: { borderColor: "#243352", timeVisible: true, secondsVisible: false },
    });
    const series = chart.addCandlestickSeries({
      upColor: "#00FF88",
      downColor: "#FF4D4D",
      borderUpColor: "#00FF88",
      borderDownColor: "#FF4D4D",
      wickUpColor: "#00FF88",
      wickDownColor: "#FF4D4D",
    });
    const vol = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "vol",
      color: "rgba(59,130,246,0.4)",
    });
    chart.priceScale("vol").applyOptions({
      scaleMargins: { top: 0.82, bottom: 0 },
    });
    chartRef.current = chart;
    seriesRef.current = series;
    volRef.current = vol;

    const ro = new ResizeObserver(() => {
      if (containerRef.current)
        chart.applyOptions({ width: containerRef.current.clientWidth });
    });
    ro.observe(containerRef.current);
    chart.applyOptions({ width: containerRef.current.clientWidth });

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [height]);

  // update data
  useEffect(() => {
    if (!seriesRef.current || !volRef.current) return;
    const data = candles.map((c) => ({
      time: c.time as Time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    seriesRef.current.setData(data);
    volRef.current.setData(
      candles.map((c) => ({
        time: c.time as Time,
        value: c.volume,
        color: c.close >= c.open ? "rgba(0,255,136,0.3)" : "rgba(255,77,77,0.3)",
      }))
    );

    // Markers. The paper trades carry a *simulation step* timestamp, not a
    // real unix epoch, so we cannot snap by time. Strategy:
    //  1) try real epoch (ISO date or seconds within the candle window);
    //  2) otherwise place each trade on the candle whose CLOSE price is
    //     nearest to the trade price (truthful: the trade happened at that
    //     price level), keeping chronological order.
    if (candles.length && trades.length) {
      const times = candles.map((c) => c.time);
      const minT = times[0];
      const maxT = times[times.length - 1];

      const epochMarker = (t: TradeRow) => {
        let ts = NaN;
        if (t.timestamp) {
          const asNum = Number(t.timestamp);
          if (!Number.isNaN(asNum) && asNum >= minT && asNum <= maxT) ts = asNum;
          else {
            const p = Math.floor(Date.parse(t.timestamp) / 1000);
            if (!Number.isNaN(p) && p >= minT && p <= maxT) ts = p;
          }
        }
        return ts;
      };

      const hasEpoch = trades.some((t) => !Number.isNaN(epochMarker(t)));

      const markers = trades
        .map((t, idx) => {
          let snap: number;
          const ep = epochMarker(t);
          if (hasEpoch && !Number.isNaN(ep)) {
            snap = ep;
          } else if (t.price && t.price > 0) {
            // nearest candle by price
            let best = times[0];
            let bd = Infinity;
            for (let i = 0; i < candles.length; i++) {
              const d = Math.abs(candles[i].close - t.price);
              if (d < bd) {
                bd = d;
                best = candles[i].time;
              }
            }
            snap = best;
          } else {
            // even distribution fallback
            const k = Math.min(
              candles.length - 1,
              Math.floor((idx / trades.length) * candles.length)
            );
            snap = times[k];
          }
          return markerFor({ time: snap, side: t.side, reason: t.reason, price: t.price });
        })
        .filter(Boolean) as ReturnType<typeof markerFor>[];
      markers.sort((a, b) => (a.time as number) - (b.time as number));
      // dedupe identical (time, shape) to avoid lib warnings
      const seen = new Set<string>();
      const deduped = markers.filter((mk) => {
        const key = `${mk.time}-${mk.shape}-${mk.text}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
      seriesRef.current.setMarkers(deduped);
    } else {
      seriesRef.current.setMarkers([]);
    }

    chartRef.current?.timeScale().fitContent();
  }, [candles, trades]);

  return <div ref={containerRef} className="w-full" style={{ height }} />;
}
