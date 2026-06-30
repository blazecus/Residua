"""
SB3-compatible VecEnv that runs N agents inside a single Residua simulation.

Usage:
    from residua_vec_env import ResiduaVecEnv
    env = ResiduaVecEnv(n_envs=16)
    obs = env.reset()                       # (16, 313)
    obs, rewards, dones, infos = env.step(actions)  # actions: (16, 13)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bin"))

import random
import numpy as np
import residua_py
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

JOINT_COUNT = 13
OBS_SIZE    = (18 + JOINT_COUNT) + JOINT_COUNT * 3  # 70: 1 frame + action + cur anim goal + next anim goal

class _MoveController:
    """
    Random left/right/stop movement input for one agent.
    Holds each state for a random duration, then picks the next.
    """

    def __init__(self):
        self.dir   = 0.
        self.timer = 0.
        self._pick_next()

    def step(self, dt: float) -> float:
        self.timer -= dt
        if self.timer <= 0.:
            self._pick_next()
        return self.dir

    def _pick_next(self):
        r = random.random()
        if r < 0.38:
            self.dir   = 1.
            self.timer = random.uniform(0.6, 3.5)
        elif r < 0.76:
            self.dir   = -1.
            self.timer = random.uniform(0.6, 3.5)
        else:
            self.dir   = 0.
            self.timer = random.uniform(0.2, 1.5)

class _AimController:
    """
    Smooth random aim position for one agent.
    Picks random waypoints and travels at varying speeds.
    Occasionally pauses at a waypoint before moving again.
    """
    _SPEED_BANDS = [
        (20.,  80.),   
        (20.,  80.),   
        (100., 250.),  
        (100., 250.),  
        (350., 700.),  
    ]

    def __init__(self, x: float, y: float,
                 x_min: float, x_max: float,
                 y_min: float, y_max: float):
        self.pos    = np.array([x, y], dtype=np.float32)
        self.target = np.array([x, y], dtype=np.float32)
        self.speed  = 0.
        self.pause  = 0.
        self._x_min = x_min; self._x_max = x_max
        self._y_min = y_min; self._y_max = y_max
        self._pick_target()

    def step(self, dt: float) -> tuple:
        if self.pause > 0.:
            self.pause = max(0., self.pause - dt)
            return float(self.pos[0]), float(self.pos[1])

        diff = self.target - self.pos
        dist = float(np.linalg.norm(diff))

        if dist < 5.:
            if random.random() < 0.4:
                self.pause = random.uniform(0.1, 1.4)
            self._pick_target()
        else:
            move = min(self.speed * dt, dist)
            self.pos += (diff / dist) * move

        return float(self.pos[0]), float(self.pos[1])

    def _pick_target(self):
        self.target = np.array([
            random.uniform(self._x_min, self._x_max),
            random.uniform(self._y_min, self._y_max),
        ], dtype=np.float32)
        lo, hi = random.choice(self._SPEED_BANDS)
        self.speed = random.uniform(lo, hi)


class ResiduaVecEnv(VecEnv):
    """
    All n_envs agents share one physics world — no inter-process overhead.
    Done agents are auto-reset in step_wait (SB3 convention).
    """

    def __init__(
        self,
        n_envs:      int   = 8,
        spawn_x:     float = 0.,
        spawn_y:     float = 270.,
        render:      bool  = False,
        scene_idx:   int   = 5,
        aim_x:       float = 100.,
        aim_y:       float = 270.,
        max_steps:   int   = 1000,
        aim_x_range: tuple = (-280., 280.),
        aim_y_range: tuple = (130., 340.),
        physics_dt:  float = 1. / 60.,
    ):
        obs_space = spaces.Box(-np.inf, np.inf, shape=(OBS_SIZE,),    dtype=np.float32)
        act_space = spaces.Box(-1.,     1.,      shape=(JOINT_COUNT,), dtype=np.float32)
        super().__init__(n_envs, obs_space, act_space)

        self._sim = residua_py.PyEnv()
        self._sim.init(n_envs, spawn_x, spawn_y, render, scene_idx)

        self._aim_xs      = [aim_x] * n_envs
        self._aim_ys      = [aim_y] * n_envs
        self._aim_ctrls   = [
            _AimController(aim_x, aim_y,
                           aim_x_range[0], aim_x_range[1],
                           aim_y_range[0], aim_y_range[1])
            for _ in range(n_envs)
        ]
        self._move_dirs   = [0.] * n_envs
        self._move_ctrls  = [_MoveController() for _ in range(n_envs)]
        self._physics_dt  = physics_dt
        self._max_steps   = max_steps
        self._step_counts = np.zeros(n_envs, dtype=np.int32)
        self._actions     = None
        self._render      = render

    def step_async(self, actions: np.ndarray):
        self._actions = actions  

    def step_wait(self):
        for i in range(self.num_envs):
            self._aim_xs[i], self._aim_ys[i] = self._aim_ctrls[i].step(self._physics_dt)
            self._move_dirs[i] = self._move_ctrls[i].step(self._physics_dt)

        obs_list, rewards, dones = self._sim.step(
            self._actions.tolist(),
            self._move_dirs,  
            [],               
            self._aim_xs,
            self._aim_ys,
            False,          
            self._render,
        )

        self._step_counts += 1
        infos = [{} for _ in range(self.num_envs)]

        for i, done in enumerate(dones):
            truncated = int(self._step_counts[i]) >= self._max_steps
            if done or truncated:
                infos[i]["terminal_observation"] = np.array(obs_list[i], dtype=np.float32)
                infos[i]["TimeLimit.truncated"]   = truncated and not done
                self._sim.reset_agent(i)
                obs_list[i] = self._sim.get_agent_obs(i)
                self._step_counts[i] = 0
                dones[i] = True

        return (
            np.array(obs_list, dtype=np.float32),
            np.array(rewards,  dtype=np.float32),
            np.array(dones,    dtype=bool),
            infos,
        )

    def reset(self):
        self._sim.reset_all()
        self._step_counts[:] = 0
        return np.array(self._sim.get_obs_all(), dtype=np.float32)

    def close(self):
        self._sim.close()

    def get_attr(self, attr_name, indices=None):
        n = self.num_envs if indices is None else len(indices)
        return [getattr(self, attr_name, None)] * n

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self, method_name)(*method_args, **method_kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        n = self.num_envs if indices is None else len(indices)
        return [False] * n

    def seed(self, seed=None):
        return [None] * self.num_envs
