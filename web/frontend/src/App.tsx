import { useState } from "react";
import Dashboard from "./tabs/Dashboard";
import Training from "./tabs/Training";
import Soon from "./tabs/Soon";
import { usePoll } from "./usePoll";
import { api } from "./api";

const TABS = [
  "Dashboard",
  "Training",
  "Backtest",
  "Paper",
  "Live",
  "Research",
  "Models",
  "Agents",
  "System",
] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [tab, setTab] = useState<Tab>("Dashboard");
  const { data: status } = usePoll(api.status, 4000);
  const running = status?.process.running;

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-edge bg-panel/80 backdrop-blur sticky top-0 z-10">
        <div className="px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-accent text-lg">◢◤</span>
            <h1 className="text-sm font-semibold tracking-widest">
              ADAN0 <span className="text-muted">TERMINAL</span>
            </h1>
            <span className="text-[10px] text-muted hidden sm:inline">
              Mission Control · BTC/USDT · scalper
            </span>
          </div>
          <div className="flex items-center gap-3">
            <a
              href="/docs"
              target="_blank"
              rel="noreferrer"
              className="text-[11px] text-info hover:underline"
            >
              API Swagger ↗
            </a>
            <span
              className={`flex items-center gap-1 text-[11px] ${
                running ? "text-up" : "text-down"
              }`}
            >
              <span className={running ? "pulse-dot" : ""}>●</span>
              {running ? "LIVE" : "OFFLINE"}
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

      <main className="flex-1 p-4 max-w-[1400px] w-full mx-auto">
        {tab === "Dashboard" && <Dashboard />}
        {tab === "Training" && <Training />}
        {tab === "Backtest" && (
          <Soon
            name="Backtest Studio"
            phase="Phase 2"
            desc="Equity curve, distribution des trades, MAE/MFE, confusion matrix, timeline des positions — wrap des scripts scripts/backtest/*."
          />
        )}
        {tab === "Paper" && (
          <Soon
            name="Paper Trading Center"
            phase="Phase 3"
            desc="Positions ouvertes, ordres, PnL et exposition simulés en temps réel."
          />
        )}
        {tab === "Live" && (
          <Soon
            name="Live Trading Center"
            phase="Phase 3"
            desc="État exchange, capital, levier, ordres, métriques de risque, alertes."
          />
        )}
        {tab === "Research" && (
          <Soon
            name="Research Lab"
            phase="Phase 2"
            desc="Campagnes scripts/research/* : confusion matrix, winner distribution, fee horizon sensitivity, zone lookahead audit."
          />
        )}
        {tab === "Models" && (
          <Soon
            name="Model Registry / Checkpoint Explorer"
            phase="Phase 2"
            desc="Checkpoints versionnés, métadonnées, promotion/déploiement, comparaison de runs."
          />
        )}
        {tab === "Agents" && (
          <Soon
            name="Agents"
            phase="Phase 3"
            desc="Communication multi-agents, tâches, statut, logs, timeouts."
          />
        )}
        {tab === "System" && (
          <Soon
            name="System Center"
            phase="Phase 3"
            desc="CPU/RAM/swap/Docker/process — au MVP, surveillé dans le Dashboard."
          />
        )}
      </main>

      <footer className="border-t border-edge px-4 py-2 text-[10px] text-muted flex justify-between">
        <span>ADAN0 v2 — Arène Guidée par le Futur</span>
        <span>read-only · frais verrouillés (0.5%)</span>
      </footer>
    </div>
  );
}
