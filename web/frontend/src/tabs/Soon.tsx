import { Panel, Badge } from "../components/ui";

export default function Soon({
  name,
  phase,
  desc,
}: {
  name: string;
  phase: string;
  desc: string;
}) {
  return (
    <Panel title={name} right={<Badge tone="info">{phase}</Badge>}>
      <p className="text-sm text-zinc-400">{desc}</p>
      <p className="text-xs text-muted mt-3">
        Spécifié dans docs/web_interface.md. La Phase 1 (MVP) se concentre sur
        le suivi temps réel de l'entraînement 500k V4.
      </p>
    </Panel>
  );
}
