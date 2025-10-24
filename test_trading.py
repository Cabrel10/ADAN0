#!/usr/bin/env python3
"""
Test rapide pour vérifier que le bot peut effectuer des trades.
"""
import yaml

# Charger la config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Vérifier les paramètres critiques
print("=" * 80)
print("🔍 VÉRIFICATION DES PARAMÈTRES CRITIQUES")
print("=" * 80)

freq_config = config.get('trading_rules', {}).get('frequency', {})
min_pos = freq_config.get('min_positions', {})
grace_period = freq_config.get('grace_period_steps', 100)
action_threshold = freq_config.get('action_threshold', 0.01)

print(f"\n✅ Frequency Gate Configuration:")
print(f"   min_positions['5m']: {min_pos.get('5m', '?')}")
print(f"   min_positions['1h']: {min_pos.get('1h', '?')}")
print(f"   min_positions['4h']: {min_pos.get('4h', '?')}")
print(f"   grace_period_steps: {grace_period}")
print(f"   action_threshold: {action_threshold}")

# Vérifier les workers
print(f"\n✅ Worker Configurations:")
for worker_id in ['w1', 'w2', 'w3', 'w4']:
    worker = config.get('workers', {}).get(worker_id, {})
    min_conf = worker.get('min_confidence', '?')
    patience = worker.get('patience_steps', '?')
    print(f"   {worker_id}: min_confidence={min_conf}, patience_steps={patience}")
    
    # Vérifier tracking_periods
    spec = worker.get('specialization', {})
    tracking = spec.get('tracking_periods', {})
    for tf in ['5m', '1h', '4h']:
        if tf in tracking:
            mts = tracking[tf].get('min_tracking_steps', '?')
            gp = tracking[tf].get('grace_period', '?')
            print(f"      {tf}: min_tracking_steps={mts}, grace_period={gp}")

# Vérifier si le bot PEUT trader
print(f"\n🎯 DIAGNOSTIC:")
all_zero = all(min_pos.get(tf, 1) == 0 for tf in ['5m', '1h', '4h'])
if all_zero:
    print(f"   ✅ FREQUENCY GATE DÉSACTIVÉ - Le bot PEUT prendre son premier trade!")
else:
    print(f"   ❌ FREQUENCY GATE ACTIF - Le bot est BLOQUÉ!")
    print(f"      Solution: Mettre tous les min_positions à 0")

if grace_period <= 10:
    print(f"   ✅ Grace period raisonnable ({grace_period} steps)")
else:
    print(f"   ⚠️ Grace period élevé ({grace_period} steps)")

if action_threshold <= 0.05:
    print(f"   ✅ Action threshold permissif ({action_threshold})")
else:
    print(f"   ⚠️ Action threshold strict ({action_threshold})")

print("\n" + "=" * 80)
print("📊 RÉSUMÉ:")
if all_zero and grace_period <= 10 and action_threshold <= 0.05:
    print("   ✅ CONFIGURATION OPTIMALE - Le bot devrait trader!")
else:
    print("   ⚠️ CONFIGURATION SOUS-OPTIMALE - Des ajustements sont nécessaires")
print("=" * 80)
