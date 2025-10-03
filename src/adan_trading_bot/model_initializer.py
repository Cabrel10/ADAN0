"""
Module d'initialisation du modèle PPO avec configuration personnalisée.
"""
import os
import yaml
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from adan_trading_bot.model.custom_cnn import CustomCNN

def load_config(config_path):
    """
    Charge la configuration depuis un fichier YAML.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_callbacks(config, eval_env=None):
    """
    Crée les callbacks pour l'entraînement.
    """
    callbacks = []

    # Callback de sauvegarde des checkpoints
    checkpoint_cb = CheckpointCallback(
        save_freq=config['training']['checkpointing']['save_freq'],
        save_path=config['training']['checkpointing']['save_path'],
        name_prefix=config['training']['checkpointing']['model_name'],
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_cb)


    # Callback d'évaluation si un environnement d'évaluation est fourni
    if eval_env is not None:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(
                config['training']['checkpointing']['save_path'],
                'best_model'
            ),
            log_path=os.path.join(
                config['paths']['logs_dir'],
                'eval_logs'
            ),
            eval_freq=10000,
            deterministic=True,
            render=False,
            n_eval_episodes=10,
        )
        callbacks.append(eval_cb)

    return callbacks

def create_model(env, config, device='auto'):
    """
    Crée et initialise le modèle PPO avec la configuration personnalisée.
    """
    # Extraire la configuration du modèle
    model_cfg = config['model']
    training_cfg = config['training']

    # Configuration de l'extracteur de caractéristiques
    features_extractor_kwargs = {
        'cnn_configs': {
            '5m': model_cfg['architecture'],  # Configuration pour le timeframe 5m
            # Ajouter d'autres timeframes si nécessaire
        },
        'diagnostics': model_cfg.get('diagnostics', {}),
    }

    # Configuration de la politique
    policy_kwargs = {
        'features_extractor_class': CustomCNN,
        'features_extractor_kwargs': features_extractor_kwargs,
        'net_arch': [
            dict(
                pi=model_cfg['architecture']['head']['hidden_units'],
                vf=model_cfg['architecture']['head']['hidden_units']
            )
        ],
        'activation_fn': torch.nn.LeakyReLU,
        'ortho_init': True,
    }

    # Créer le modèle PPO
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=training_cfg.get('learning_rate', 3e-4),
        n_steps=training_cfg.get('n_steps', 2048),
        batch_size=training_cfg.get('batch_size', 64),
        n_epochs=training_cfg.get('n_epochs', 10),
        gamma=training_cfg.get('gamma', 0.99),
        gae_lambda=training_cfg.get('gae_lambda', 0.95),
        clip_range=training_cfg.get('clip_range', 0.2),
        clip_range_vf=training_cfg.get('clip_range_vf', None),
        ent_coef=training_cfg.get('ent_coef', 0.0),
        vf_coef=training_cfg.get('vf_coef', 0.5),
        max_grad_norm=training_cfg.get('max_grad_norm', 0.5),
        use_sde=training_cfg.get('use_sde', False),
        sde_sample_freq=training_cfg.get('sde_sample_freq', -1),
        target_kl=training_cfg.get('target_kl', None),
        tensorboard_log=os.path.join(config['paths']['logs_dir'], 'tensorboard'),
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        seed=config['general']['random_seed'],
    )

    return model

def setup_training(config_path, env, eval_env=None, device='auto'):
    """
    Configure l'entraînement complet avec les callbacks et le modèle.
    """
    # Charger la configuration
    config = load_config(config_path)

    # Créer le modèle
    model = create_model(env, config, device)

    # Créer les callbacks
    callbacks = create_callbacks(config, eval_env)

    return model, callbacks, config

def train_model(model, total_timesteps, callbacks=None, progress_bar=True):
    """
    Lance l'entraînement du modèle.
    """
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=progress_bar,
        reset_num_timesteps=True,
    )

    return model
