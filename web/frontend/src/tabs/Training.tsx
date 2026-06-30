import { useEffect, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
} from "recharts";
import { Panel, Badge } from "../components/ui";
import { usePoll } from "../usePoll";
import { api, TelemetryRow } from "../api";

const AXIS = { stroke: "#64748B", fontSize: 10 };
const GRID = "rgba(36,51,82,0.5)";

function tooltipStyle() {
  return {
    contentStyle: {
      background: "#111B2E",
      border: "1px solid #243352",
      borderRadius: 6,
      fontSize: 11,
    },
    labelStyle: { color: "#CBD5E1" },
  };
}

export default function Training() {
  const [rows, setRows] = useState<TelemetryRow[]>([]);
  const sinceRef = useRef(0);

  const { data: tele } = usePoll(
    () => api.telemetry(sinceRef.current),
    4000
  );

  useEffect(() => {
    if (tele?.rows?.length) {
      setRows((prev) => {
        const merged = [...prev, ...tele.rows];
        sinceRef.current = merged[merged.length - 1].timesteps;
        return merged.slice(-400);
      });
    }
  }, [tele]);

  const { data: collapse } = usePoll(api.collapse, 5000);
  const { data: logd } = usePoll(() => api.log(200), 4000);

  const last = rows[rows.length - 1];
  const histoData =
    last?.a0_histo?.map((v, i) => ({ bucket: `b${i}`, count: v })) ?? [];

  return (
    <div className="space-y-4">
      <Panel
        title="Détection de collapse (FAITS)"
        right={
          <Badge
            tone={
              collapse?.level === "critical"
                ? "crit"
                : collapse?.level === "warning"
                ? "warn"
                : "ok"
            }
          >
            {collapse?.status ?? "…"}
          </Badge>
        }
      >
        <div className="text-xs text-zinc-300 flex flex-wrap gap-x-6 gap-y-1">
          {(collapse?.reasons ?? []).map((r, i) => (
            <span key={i}>› {r}</span>
          ))}
          {rows.length === 0 && (
            <span className="text-muted">
              En attente du premier flush télémétrie (tous les 5000 steps)…
            </span>
          )}
        </div>
      </Panel>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Panel title="action0 — std & entropy">
          <Chart>
            <LineChart data={rows}>
              <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
              <XAxis dataKey="timesteps" {...AXIS} />
              <YAxis {...AXIS} />
              <Tooltip {...tooltipStyle()} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line
                type="monotone"
                dataKey="a0_std"
                stroke="#FF4D4D"
                dot={false}
                name="a0_std"
              />
              <Line
                type="monotone"
                dataKey="policy_entropy"
                stroke="#8B5CF6"
                strokeWidth={1.5}
                dot={false}
                name="entropy"
              />
            </LineChart>
          </Chart>
        </Panel>

        <Panel title="Actions demandées (%)">
          <Chart>
            <LineChart data={rows}>
              <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
              <XAxis dataKey="timesteps" {...AXIS} />
              <YAxis {...AXIS} domain={[0, 1]} />
              <Tooltip {...tooltipStyle()} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="req_HOLD_pct" stroke="#3B82F6" dot={false} name="HOLD" />
              <Line type="monotone" dataKey="req_BUY_pct" stroke="#00FF88" dot={false} name="BUY" />
              <Line type="monotone" dataKey="req_SELL_pct" stroke="#FF4D4D" dot={false} name="SELL" />
            </LineChart>
          </Chart>
        </Panel>

        <Panel title="illegal_ratio & flat/open">
          <Chart>
            <LineChart data={rows}>
              <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
              <XAxis dataKey="timesteps" {...AXIS} />
              <YAxis {...AXIS} domain={[0, 1]} />
              <Tooltip {...tooltipStyle()} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="illegal_ratio" stroke="#FFC857" dot={false} name="illegal" />
              <Line type="monotone" dataKey="steps_flat_pct" stroke="#64748B" dot={false} name="flat%" />
              <Line type="monotone" dataKey="steps_open_pct" stroke="#00FF88" dot={false} name="open%" />
            </LineChart>
          </Chart>
        </Panel>

        <Panel title={`Histogramme action0 ${last ? `@${last.timesteps}` : ""}`}>
          <Chart>
            <BarChart data={histoData}>
              <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
              <XAxis dataKey="bucket" {...AXIS} />
              <YAxis {...AXIS} />
              <Tooltip {...tooltipStyle()} />
              <Bar dataKey="count" fill="#3B82F6" />
            </BarChart>
          </Chart>
          <p className="text-[10px] text-muted mt-1">
            Collapse = masse concentrée aux extrémités (b0 / b9). Sain = réparti.
          </p>
        </Panel>
      </div>

      <Panel title="Console live — train_v4_500k.log">
        <pre className="text-[10px] leading-relaxed max-h-72 overflow-auto bg-black/40 rounded p-2">
          {(logd?.lines ?? ["(en attente du log…)"]).slice(-120).map((l, i) => (
            <div
              key={i}
              className={
                /error|nameerror|traceback|not defined/i.test(l)
                  ? "text-down"
                  : /checkpoint|saved/i.test(l)
                  ? "text-up"
                  : /sterile/i.test(l)
                  ? "text-warn"
                  : "text-zinc-400"
              }
            >
              {l}
            </div>
          ))}
        </pre>
      </Panel>
    </div>
  );
}

function Chart({ children }: { children: any }) {
  return (
    <div style={{ width: "100%", height: 220 }}>
      <ResponsiveContainer>{children}</ResponsiveContainer>
    </div>
  );
}
