import { useEffect, useState } from "react";
import { Panel, Badge } from "../components/ui";
import { api, ControlStatus } from "../api";

// All editable hyperparams shown in the UI. Only those whitelisted by the
// backend (editable_hyperparams) are actually sent; the rest are display-only
// hints so the user sees the full PPO surface. FEES are never editable.
const HP_FIELDS: { key: string; label: string; def: number; step: number }[] = [
  { key: "learning_rate", label: "learning_rate", def: 0.0003, step: 0.0001 },
  { key: "gamma", label: "gamma", def: 0.99, step: 0.001 },
  { key: "gae_lambda", label: "gae_lambda", def: 0.95, step: 0.01 },
  { key: "clip_range", label: "clip_range", def: 0.2, step: 0.01 },
  { key: "ent_coef", label: "ent_coef", def: 0.03, step: 0.005 },
  { key: "vf_coef", label: "vf_coef", def: 0.5, step: 0.05 },
  { key: "batch_size", label: "batch_size", def: 256, step: 64 },
  { key: "n_steps", label: "n_steps", def: 2048, step: 256 },
];

export default function HyperLab() {
  const [ctrl, setCtrl] = useState<ControlStatus | null>(null);
  const [worker, setWorker] = useState("scalper");
  const [steps, setSteps] = useState(50000);
  const [diag, setDiag] = useState(true);
  const [hp, setHp] = useState<Record<string, number>>(
    Object.fromEntries(HP_FIELDS.map((f) => [f.key, f.def]))
  );
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = () => api.controlStatus().then(setCtrl).catch(() => {});
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);

  const editable = new Set(ctrl?.editable_hyperparams ?? []);
  const running = ctrl?.status.running;

  const launch = async () => {
    setBusy(true);
    setMsg(null);
    // only send whitelisted hyperparams to the backend
    const payload: Record<string, number> = {};
    for (const k of Object.keys(hp)) if (editable.has(k)) payload[k] = hp[k];
    try {
      const r = await api.launch({ steps, worker, hyperparams: payload, diag });
      setMsg({ ok: r.ok, text: r.message || (r.ok ? `lancé (pid ${r.pid})` : "échec") });
    } catch (e) {
      setMsg({ ok: false, text: String(e) });
    } finally {
      setBusy(false);
      refresh();
    }
  };

  const stop = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const r = await api.stop();
      setMsg({ ok: r.ok, text: r.message || "stop envoyé" });
    } catch (e) {
      setMsg({ ok: false, text: String(e) });
    } finally {
      setBusy(false);
      refresh();
    }
  };

  return (
    <div className="space-y-4">
      <Panel
        title="Worker Management"
        right={
          <span className={`text-[11px] ${running ? "text-up" : "text-muted"}`}>
            {running ? `● actif (pid ${ctrl?.status.pid})` : "○ aucun run"}
          </span>
        }
      >
        <div className="flex flex-wrap gap-2 mb-3">
          {(ctrl?.workers ?? ["scalper", "w1", "w2", "w3"]).map((w) => (
            <button
              key={w}
              onClick={() => setWorker(w)}
              className={`text-xs px-3 py-1.5 rounded border ${
                worker === w
                  ? "border-accent text-zinc-100 bg-accent/15"
                  : "border-edge text-muted hover:text-zinc-300"
              }`}
            >
              {w}
            </button>
          ))}
          <button
            onClick={() => setWorker("all")}
            className={`text-xs px-3 py-1.5 rounded border ${
              worker === "all"
                ? "border-info text-zinc-100 bg-info/15"
                : "border-edge text-muted hover:text-zinc-300"
            }`}
          >
            train all
          </button>
        </div>
        <div className="flex flex-wrap items-center gap-4 text-xs">
          <label className="flex items-center gap-2">
            <span className="text-muted">steps</span>
            <input
              type="number"
              value={steps}
              min={1000}
              step={10000}
              onChange={(e) => setSteps(Number(e.target.value))}
              className="bg-panel2 border border-edge rounded px-2 py-1 w-28 font-mono"
            />
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={diag}
              onChange={(e) => setDiag(e.target.checked)}
              className="accent-violet-500"
            />
            <span className="text-muted">diagnostic collapse</span>
          </label>
        </div>
      </Panel>

      <Panel
        title="Hyperparameter Lab"
        right={<Badge tone="warn">frais 0.5% verrouillés</Badge>}
      >
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          {HP_FIELDS.map((f) => {
            const isEditable = editable.has(f.key);
            return (
              <div
                key={f.key}
                className={`bg-panel2 border rounded-md px-3 py-2 ${
                  isEditable ? "border-edge" : "border-edge/40 opacity-60"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[10px] uppercase tracking-wide text-muted">
                    {f.label}
                  </span>
                  {isEditable ? (
                    <span className="text-[9px] text-up">live</span>
                  ) : (
                    <span className="text-[9px] text-muted">config</span>
                  )}
                </div>
                <input
                  type="number"
                  value={hp[f.key]}
                  step={f.step}
                  disabled={!isEditable}
                  onChange={(e) =>
                    setHp((s) => ({ ...s, [f.key]: Number(e.target.value) }))
                  }
                  className="bg-bg border border-edge rounded px-2 py-1 w-full mt-1 font-mono text-sm disabled:opacity-50"
                />
              </div>
            );
          })}
        </div>
        <div className="text-[10px] text-muted mt-3">
          Les champs <span className="text-up">live</span> sont injectés via
          variables d'environnement au lancement (jamais via réécriture de{" "}
          <code>config.yaml</code>). Les autres servent de référence. Clés
          interdites : {(ctrl?.forbidden ?? ["commission", "round_trip_fees", "fee", "fees"]).join(", ")}.
        </div>
      </Panel>

      <Panel title="Contrôle du run">
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={launch}
            disabled={busy || running}
            className="text-sm px-4 py-2 rounded bg-up/15 border border-up/50 text-up hover:bg-up/25 disabled:opacity-40"
          >
            ▶ Launch
          </button>
          <button
            onClick={stop}
            disabled={busy || !running}
            className="text-sm px-4 py-2 rounded bg-down/15 border border-down/50 text-down hover:bg-down/25 disabled:opacity-40"
          >
            ■ Stop
          </button>
          <span className="text-[11px] text-muted">
            worker=<b className="text-zinc-100">{worker}</b> · steps=
            <b className="text-zinc-100">{steps.toLocaleString()}</b>
          </span>
        </div>
        {msg && (
          <div className={`text-xs mt-3 ${msg.ok ? "text-up" : "text-down"}`}>
            {msg.text}
          </div>
        )}
      </Panel>
    </div>
  );
}
