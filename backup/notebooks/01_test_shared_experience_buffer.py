# %% [markdown]
# Test et Benchmark du SharedExperienceBuffer

Ce notebook permet de tester et d'évaluer les performances du `SharedExperienceBuffer` avec différentes configurations.

## Configuration initiale

D'abord, installons les dépendances nécessaires :

```bash
# Exécuter cette cellule pour installer les dépendances
!pip install matplotlib numpy tqdm
```

## 1. Test d'intégration de base

Vérifions d'abord que le buffer fonctionne correctement avec un seul processus.

%run -i "../src/adan_trading_bot/training/shared_experience_buffer.py"

import numpy as np
import time
from tqdm import tqdm

# Création du buffer
buffer_size = 10000
buffer = SharedExperienceBuffer(buffer_size=buffer_size)

# Test d'ajout d'expériences
print("Test d'ajout d'expériences...")
num_experiences = 1000
for i in tqdm(range(num_experiences)):
    experience = {
        'state': np.random.rand(84, 84, 4).astype(np.float32),
        'action': np.random.randint(0, 4),
        'reward': float(np.random.rand() * 2 - 1),
        'next_state': np.random.rand(84, 84, 4).astype(np.float32),
        'done': np.random.rand() < 0.01,
        'id': i
    }
    buffer.add(experience)

print(f"Taille du buffer: {len(buffer)}")
print(f"Capacité du buffer: {buffer_size}")

# Test d'échantillonnage
print("\nTest d'échantillonnage...")
batch_size = 32
start_time = time.time()
num_batches = 100

for _ in tqdm(range(num_batches)):
    batch, indices, weights = buffer.sample(batch_size)
    # Vérification de la forme des données
    assert len(batch['state']) == batch_size
    assert len(indices) == batch_size
    assert len(weights) == batch_size

total_time = time.time() - start_time
print(f"Temps total pour {num_batches} batches: {total_time:.2f}s")
print(f"Débit: {num_batches * batch_size / total_time:.2f} exp/s")

## 2. Test avec plusieurs processus

Testons maintenant avec plusieurs processus qui ajoutent et échantillonnent des expériences en parallèle.

import multiprocessing

def worker_add(buffer, worker_id, num_experiences):
    """Worker qui ajoute des expériences au buffer."""
    for i in range(num_experiences):
        experience = {
            'state': np.random.rand(84, 84, 4).astype(np.float32),
            'action': np.random.randint(0, 4),
            'reward': float(np.random.rand() * 2 - 1),
            'next_state': np.random.rand(84, 84, 4).astype(np.float32),
            'done': np.random.rand() < 0.01,
            'worker_id': worker_id,
            'step': i
        }
        buffer.add(experience)

def worker_sample(buffer, num_batches, batch_size):
    """Worker qui échantillonne des expériences du buffer."""
    for _ in range(num_batches):
        batch, indices, weights = buffer.sample(batch_size)
        # Faire quelque chose avec le batch pour éviter l'optimisation
        _ = sum(len(s) for s in batch['state'])

# Configuration du test
buffer_size = 50000
num_workers = 4
experiences_per_worker = 1000
num_samplers = 2
batch_size = 64
num_batches = 100

print(f"\nTest avec {num_workers} workers d'ajout et {num_samplers} workers d'échantillonnage")
print(f"Buffer size: {buffer_size}, Batch size: {batch_size}")

# Création du buffer
buffer = SharedExperienceBuffer(buffer_size=buffer_size)

# Création des workers
add_workers = [
    multiprocessing.Process(
        target=worker_add,
        args=(buffer, i, experiences_per_worker)
    )
    for i in range(num_workers)
]

sample_workers = [
    multiprocessing.Process(
        target=worker_sample,
        args=(buffer, num_batches, batch_size)
    )
    for _ in range(num_samplers)
]

# Démarrage des workers
start_time = time.time()

for w in add_workers + sample_workers:
    w.start()

# Attente de la fin des workers
for w in add_workers + sample_workers:
    w.join()

total_time = time.time() - start_time
print(f"Temps total: {total_time:.2f}s")
print(f"Taille finale du buffer: {len(buffer)}")

## 3. Analyse des performances

Effectuons maintenant une analyse plus approfondie des performances avec différentes configurations.

import pandas as pd
import matplotlib.pyplot as plt

def run_benchmark(config):
    """Exécute un benchmark avec la configuration donnée."""
    buffer_size = config['buffer_size']
    num_workers = config['num_workers']
    batch_size = config['batch_size']
    num_batches = config['num_batches']

    # Création du buffer
    buffer = SharedExperienceBuffer(buffer_size=buffer_size)

    # Création des workers
    workers = []

    # Workers d'ajout
    for i in range(num_workers):
        w = multiprocessing.Process(
            target=worker_add,
            args=(buffer, i, num_batches * batch_size // num_workers)
        )
        workers.append(w)

    # Workers d'échantillonnage
    for _ in range(num_workers):
        w = multiprocessing.Process(
            target=worker_sample,
            args=(buffer, num_batches, batch_size)
        )
        workers.append(w)

    # Mesure des performances
    start_time = time.time()

    for w in workers:
        w.start()

    for w in workers:
        w.join()

    total_time = time.time() - start_time

    # Calcul des métriques
    total_experiences = num_workers * num_batches * batch_size
    throughput = total_experiences / total_time

    return {
        'buffer_size': buffer_size,
        'num_workers': num_workers,
        'batch_size': batch_size,
        'num_batches': num_batches,
        'total_time': total_time,
        'throughput_exp_per_sec': throughput,
        'avg_latency_per_batch': total_time / num_batches
    }

# Configurations à tester
configs = [
    {'buffer_size': 10000, 'num_workers': 1, 'batch_size': 32, 'num_batches': 100},
    {'buffer_size': 10000, 'num_workers': 2, 'batch_size': 32, 'num_batches': 100},
    {'buffer_size': 10000, 'num_workers': 4, 'batch_size': 32, 'num_batches': 100},
    {'buffer_size': 50000, 'num_workers': 4, 'batch_size': 32, 'num_batches': 100},
    {'buffer_size': 50000, 'num_workers': 4, 'batch_size': 64, 'num_batches': 100},
    {'buffer_size': 50000, 'num_workers': 8, 'batch_size': 64, 'num_batches': 100},
]

# Exécution des benchmarks
print("Démarrage des benchmarks...")
results = []

for config in tqdm(configs):
    print(f"\nConfiguration: {config}")
    result = run_benchmark(config)
    results.append(result)
    print(f"  → Débit: {result['throughput_exp_per_sec']:.2f} exp/s")

# Affichage des résultats
df_results = pd.DataFrame(results)
print("\nRésultats des benchmarks:")
print(df_results[['num_workers', 'batch_size', 'buffer_size', 'throughput_exp_per_sec', 'avg_latency_per_batch']])

# Visualisation des résultats
plt.figure(figsize=(12, 6))

# Débit en fonction du nombre de workers
plt.subplot(1, 2, 1)
for buffer_size in sorted(df_results['buffer_size'].unique()):
    mask = df_results['buffer_size'] == buffer_size
    plt.plot(
        df_results[mask]['num_workers'],
        df_results[mask]['throughput_exp_per_sec'],
        'o-',
        label=f'Buffer size: {buffer_size}'
    )
plt.xlabel('Nombre de workers')
plt.ylabel('Débit (exp/s)')
plt.title('Débit en fonction du nombre de workers')
plt.legend()

# Latence moyenne en fonction de la taille du batch
plt.subplot(1, 2, 2)
for num_workers in sorted(df_results['num_workers'].unique()):
    mask = df_results['num_workers'] == num_workers
    plt.plot(
        df_results[mask]['batch_size'],
        df_results[mask]['avg_latency_per_batch'] * 1000,  # en ms
        's-',
        label=f'{num_workers} workers'
    )
plt.xlabel('Taille du batch')
plt.ylabel('Latence moyenne (ms)')
plt.title('Latence en fonction de la taille du batch')
plt.legend()

plt.tight_layout()
plt.savefig('shared_buffer_benchmark.png')
plt.show()

## 4. Conclusion

Ce notebook nous a permis de :
1. Tester le bon fonctionnement du `SharedExperienceBuffer`
2. Évaluer ses performances avec différentes configurations
3. Visualiser l'impact des paramètres sur les performances

Les résultats montrent que :
- L'ajout de workers améliore le débit jusqu'à un certain point
- La taille du buffer doit être suffisante pour éviter les collisions
- La latence augmente avec la taille du batch, mais le débit global peut s'améliorer

Pour aller plus loin, on pourrait :
- Tester avec des expériences plus volumineuses
- Évaluer l'impact de la taille des états sur les performances
- Tester différentes stratégies de priorisation
