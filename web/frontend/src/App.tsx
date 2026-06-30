import { useState } from "react";
import Dashboard from "./tabs/Dashboard";
import Training from "./tabs/Training";
import TradeViz from "./tabs/TradeViz";
import Metrics from "./tabs/Metrics";
import Reliability from "./tabs/Reliability";
import Models from "./tabs/Models";
import HyperLab from "./tabs/HyperLab";
import Backtest from "./tabs/Backtest";
import Paper from "./tabs/Paper";
import Soon from "./tabs/Soon";
import { usePoll } from "./usePoll";
import { api } from "./api";

const TABS = [
  "Dashboard",
  "Training",
  "Trades",
  "Metrics",
  "Validator",
  "Models",
  "Hyper Lab",
  "Backtest",
  "Paper",
  "Live",
  "Settings",
] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [tab, setTab] = useState<Tab>("Dashboard");
  const { data: status } = usePoll(api.status, 4000);
  const { data: collapse } = usePoll(api.collapse, 6000);
  const running = status?.process.running;
  const danger = collapse?.level === "critical";

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-edge bg-panel/80 backdrop-blur sticky top-0 z-20">
        <div className="px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-accent text-lg">◢◤</span>
            <h1 className="text-sm font-semibold tracking-widest">
              ADAN <span className="text-info">MISSION</span>{" "}
              <span className="text-muted">CONTROL</span>
            </h1>
            <span className="text-[10px] text-muted hidden sm:inline">
              BTC/USDT · SPOT · scalper · lev 1
            </span>
          </div>
          <div className="flex items-center gap-3">
            {danger && (
              <span className="text-[11px] text-down border border-down/40 bg-down/10 rounded-full px-2 py-0.5 animate-pulse">
                ⚠ COLLAPSE
              </span>
            )}
            <a
              href="/docs"
              target="_blank"
              rel="noreferrer"
              className="text-[11px] text-info hover:underline"
            >
              API ↗
            </a>
            <span
              className={`flex items-center gap-1 text-[11px] ${
                running ? "text-up" : "text-down"
              }`}
            >
              <span className={running ? "pulse-dot" : ""}>●</span>
              {running ? "TRAINING" : "OFFLINE"}
            </span>
          </div>
        </div>
        <nav className="px-2 flex gap-1 overflow-x-auto">
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-3 py-2 text-xs whitespace-nowrap border-b-2 transition-colors ${
                tab === t
                  ? "border-accent text-zinc-100"
                  : "border-transparent text-muted hover:text-zinc-300"
              }`}
            >
              {t}
            </button>
          ))}
        </nav>
      </header>

      <main className="flex-1 p-4 max-w-[1500px] w-full mx-auto">
        {tab === "Dashboard" && <Dashboard />}
        {tab === "Training" && <Training />}
        {tab === "Trades" && <TradeViz />}
        {tab === "Metrics" && <Metrics />}
        {tab === "Models" && <Models />}
        {tab === "Hyper Lab" && <HyperLab />}
        {tab === "Backtest" && <Backtest />}
        {tab === "Paper" && <Paper />}
        {tab === "Live" && (
          <Soon
            name="Live Trading Center"
            phase="Phase 3"
            desc="État exchange, capital, levier, ordres, métriques de risque, panic button. Désactivé tant qu'aucune clé exchange n'est branchée."
          />
        )}
        {tab === "Settings" && (
          <Soon
            name="Settings & Notifications"
            phase="Phase 3"
            desc="Telegram / Discord / Email, thèmes, seuils d'alerte collapse. Les FRAIS (0.5%) sont verrouillés et non éditables."
          />
        )}
      </main>

      <footer className="border-t border-edge px-4 py-2 text-[10px] text-muted flex justify-between">
        <span>ADAN0 v2 — Arène Guidée par le Futur</span>
        <span>frais verrouillés 0.5% · données réelles</span>
      </footer>
    </div>
  );
}
