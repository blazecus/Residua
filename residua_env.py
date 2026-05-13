"""
Gymnasium environment wrapping the Residua physics simulation.

Usage:
    from residua_env import ResiduaEnv
    env = ResiduaEnv(render_mode="human")  # or "none" for fast training
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

import sys
import os

# The compiled extension lands in bin/ next to the exe.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bin"))

import gymnasium as gym
import numpy as np
import residua_py

OBS_SIZE   = 28
LIMB_COUNT = 14
TORQUE_MAX = 1000.0


class ResiduaEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        render_mode: str = "none",
        spawn_x: float = 0.0,
        spawn_y: float = 270.0,
        aim_x:   float = 200.0,
        aim_y:   float = 270.0,
        scene_idx: int = 5,
        max_episode_steps: int = 1000,
    ):
        super().__init__()
        assert render_mode in self.metadata["render_modes"], f"bad render_mode {render_mode!r}"

        self.render_mode       = render_mode
        self.spawn_x           = spawn_x
        self.spawn_y           = spawn_y
        self.aim_x             = aim_x
        self.aim_y             = aim_y
        self.scene_idx         = scene_idx
        self.max_episode_steps = max_episode_steps

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-TORQUE_MAX, high=TORQUE_MAX, shape=(LIMB_COUNT,), dtype=np.float32
        )

        self._env         = residua_py.PyEnv()
        self._initialized = False
        self._step_count  = 0

    # ── gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        if not self._initialized:
            self._env.init(
                self.spawn_x, self.spawn_y,
                render    = (self.render_mode == "human"),
                scene_idx = self.scene_idx,
            )
            self._initialized = True
        else:
            self._env.reset(self.spawn_x, self.spawn_y)

        # One zero-torque step to populate the observation buffer.
        obs, _, _ = self._env.step(
            [0.0] * LIMB_COUNT,
            self.aim_x, self.aim_y,
            render=(self.render_mode == "human"),
        )
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        render = self.render_mode == "human"
        obs, reward, done = self._env.step(
            action.tolist(), self.aim_x, self.aim_y, render=render
        )
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps
        return (
            np.array(obs, dtype=np.float32),
            float(reward),
            bool(done),
            truncated,
            {},
        )

    def render(self):
        pass  # rendering is driven per-step via render_mode

    def close(self):
        if self._initialized:
            self._env.close()
            self._initialized = False
