"""Actor-only model for MimicKit TEST mode.

Builds the *same* actor submodules as PPOModel/AMPModel, without
constructing critic or discriminator networks. The checkpoint key names
are intentionally kept identical to those from AMPAgent.state_dict().
"""

import learning.ppo_model as ppo_model


class InferenceModel(ppo_model.PPOModel):
    def _build_nets(self, config, env):
        # Reuse the original actor builder verbatim for weight compatibility.
        self._build_actor(config, env)

    def eval_critic(self, obs):
        raise RuntimeError("InferenceModel does not include a critic")

    def get_critic_params(self):
        return []
