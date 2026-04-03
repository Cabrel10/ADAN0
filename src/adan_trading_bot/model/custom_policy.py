from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor


class MultiInputLstmPolicy(RecurrentActorCriticPolicy):
    """
    Recurrent policy for PPO with LSTM to handle Dict observations.
    This is a simple wrapper around RecurrentActorCriticPolicy that forces the
    use of CombinedExtractor.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=CombinedExtractor,
            **kwargs,
        )
