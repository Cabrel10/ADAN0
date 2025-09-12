import gym  # noqa: F401

class SB3GymCompatibilityWrapper(gym.Wrapper):
    """
    Rend un environnement Gym >=0.26 compatible avec SB3.

    - reset() retourne obs (et non (obs, info))
    - step() retourne (obs, reward, done, info) en fusionnant
      terminated+truncated si nécessaire
    """

    def reset(self, **kwargs):
        """Réinitialise l'environnement et retourne l'observation initiale."""
        # La nouvelle API de Gymnasium retourne un tuple (obs, info)
        out = super().reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            # **CORRECTION : Ne retourner que l'observation**
            # SB3 s'attend à recevoir uniquement l'observation de la méthode reset.
            # L'objet 'info' est géré séparément par les wrappers de SB3.
            return obs
        # Pour les anciens environnements ou si le format est déjà correct
        return out

    def step(self, action):
        """Exécute une action dans l'environnement."""
        # La nouvelle API retourne : (obs, reward, terminated, truncated, info)
        # L'ancienne API attend : (obs, reward, done, info)
        out = super().step(action)
        if isinstance(out, tuple):
            if len(out) == 5:
                obs, reward, terminated, truncated, info = out
                done = bool(terminated or truncated)
                return obs, reward, done, info
            elif len(out) == 4:
                # Déjà dans l'ancien format
                return out
        return out
