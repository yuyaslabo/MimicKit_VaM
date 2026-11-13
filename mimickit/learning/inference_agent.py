"""Inference-only MimicKit agent for existing AMP checkpoints.

Only the Actor, observation normalizer and action normalizer are loaded.
No optimizer, critic, discriminator, experience buffer or learning path.
DeepMimic reference-motion observations are provided unchanged by the env.
"""

import math

import gymnasium.spaces as spaces
import numpy as np
import torch

import envs.base_env as base_env
import learning.base_agent as base_agent
import learning.inference_model as inference_model
import learning.normalizer as normalizer
import learning.return_tracker as return_tracker
import util.mp_util as mp_util
import util.torch_util as torch_util
from util.logger import Logger


class InferenceAgent(torch.nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        self._env = env
        self._device = device
        self._config = config
        self._mode = base_agent.AgentMode.TEST
        self._curr_obs = None
        self._curr_info = None

        obs_space = env.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
        self._obs_norm = normalizer.Normalizer(
            obs_space.shape, clip=10.0, device=device, dtype=obs_dtype
        )
        # Call the *original* builder to guarantee action scaling is identical.
        self._a_norm = base_agent.BaseAgent._build_action_normalizer(self)
        self._model = inference_model.InferenceModel(config['model'], env)
        self.to(device)
        self.eval()
        self._test_return_tracker = return_tracker.ReturnTracker(env.get_num_envs(), device)

    def get_num_envs(self):
        return self._env.get_num_envs()

    def calc_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def load(self, in_file):
        """Strictly load required AMP checkpoint tensors; ignore training heads."""
        try:
            saved = torch.load(in_file, map_location=self._device, weights_only=True)
        except TypeError:  # Compatibility with older PyTorch releases.
            saved = torch.load(in_file, map_location=self._device)
        if not isinstance(saved, dict):
            raise TypeError('Expected AMPAgent.state_dict() checkpoint dictionary')
        needed = self.state_dict()
        missing = [key for key in needed if key not in saved]
        mismatched = [(key, tuple(needed[key].shape), tuple(saved[key].shape))
                      for key in needed if key in saved and needed[key].shape != saved[key].shape]
        if missing or mismatched:
            raise RuntimeError(
                'Incompatible AMP checkpoint (Actor/Normalizer): '
                f'missing={missing[:12]}, shape_mismatches={mismatched[:12]}'
            )
        self.load_state_dict({key: saved[key] for key in needed}, strict=True)
        self.eval()
        Logger.print('Loaded actor + obs/action normalizers from {:s}'.format(in_file))

    def set_mode(self, mode):
        if mode != base_agent.AgentMode.TEST:
            raise ValueError('InferenceAgent supports TEST mode only')
        self._mode = mode
        self._env.set_mode(base_env.EnvMode.TEST)

    def _decide_action(self, obs, info):
        if obs.shape[-1] != self._obs_norm.get_mean().shape[-1]:
            raise ValueError('Actor observation dimension does not match checkpoint normalizer')
        norm_obs = self._obs_norm.normalize(obs)
        norm_action_dist = self._model.eval_actor(norm_obs)
        norm_action = norm_action_dist.mode  # Match PPOAgent TEST behavior.
        action = self._a_norm.unnormalize(norm_action)
        return action, {}

    def test_model(self, num_episodes):
        """Keep the original TEST loop/termination semantics and return metrics."""
        self.eval()
        self.set_mode(base_agent.AgentMode.TEST)
        num_procs = mp_util.get_num_procs()
        per_proc = int(math.ceil(num_episodes / num_procs))
        self._test_return_tracker.reset()

        with torch.no_grad():
            self._curr_obs, self._curr_info = self._env.reset()
            if per_proc == 0:
                return {'mean_return': 0.0, 'mean_ep_len': 0.0, 'num_eps': 0}
            num_envs = self.get_num_envs()
            min_eps_per_env = int(math.ceil(per_proc / num_envs))
            while True:
                action, _ = self._decide_action(self._curr_obs, self._curr_info)
                next_obs, reward, done, next_info = self._env.step(action)
                self._test_return_tracker.update(reward, done)
                done_ids = (done != base_env.DoneFlags.NULL.value).nonzero(as_tuple=False).flatten()
                self._curr_obs, self._curr_info = self._env.reset(done_ids)
                if torch.all(self._test_return_tracker.get_eps_per_env() > min_eps_per_env - 1):
                    break

        return {
            'mean_return': self._test_return_tracker.get_mean_return().item(),
            'mean_ep_len': self._test_return_tracker.get_mean_ep_len().item(),
            'num_eps': self._test_return_tracker.get_episodes(),
        }

    def train_model(self, *args, **kwargs):
        raise RuntimeError('InferenceAgent is TEST-only; training is unsupported')
