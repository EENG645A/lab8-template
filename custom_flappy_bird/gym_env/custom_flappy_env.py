"""Custom gymnasium env that inherits from FlappyBirdEnv"""

from typing import Dict, Tuple
import numpy as np
# import gymnasium as gym
from numpy import ndarray
from flappy_bird_gymnasium import FlappyBirdEnv
from flappy_bird_gymnasium.envs.flappy_bird_env import Actions
import pygame


class CustomFlappyBirdEnv(FlappyBirdEnv):
    """
    Inherits a FlappyBirdEnv 0.4.0 for customization
    Uses the 0.4.0 version on pip:
    https://github.com/markub3327/flappy-bird-gymnasium/tree/v0.4.0
    """

    def __init__(
            self,
            env_config={},
            screen_size: Tuple[int] = (288, 512),
            audio_on: bool = False,
            normalize_obs: bool = True,
            use_lidar: bool = False,
            pipe_gap: int = 100,
            bird_color: str = "yellow",
            pipe_color: str = "green",
            render_mode: str | None = None,
            background: str | None = "day",
            score_limit: int | None = None,
            debug: bool = False
            ) -> None:
        """
        env_config dict may be used to overwrite arguments.
        use_lidar has its default changed to False.
        """

        # This enables env_configs passed through
        # if using ray to overwrite  __init__ args
        screen_size = env_config.get('screen_size', screen_size)
        audio_on = env_config.get('audio_on', audio_on)
        normalize_obs = env_config.get('normalize_obs', normalize_obs)
        use_lidar = env_config.get('use_lidar', use_lidar)
        pipe_gap = env_config.get('pipe_gap', pipe_gap)
        bird_color = env_config.get('bird_color', bird_color)
        pipe_color = env_config.get('pipe_color', pipe_color)
        render_mode = env_config.get('render_mode', render_mode)
        background = env_config.get('background', background)
        score_limit = env_config.get('score_limit', score_limit)
        debug = env_config.get('debug', debug)
        super().__init__(
            screen_size,
            audio_on,
            normalize_obs,
            use_lidar,
            pipe_gap,
            bird_color,
            pipe_color,
            render_mode,
            background,
            score_limit,
            debug
        )

    def step(
            self,
            action: Actions | int
            ) -> Tuple[ndarray | float | bool | Dict]:

        obs, reward, terminal, truncated, info = super().step(action)

        return (
            obs,
            reward,
            terminal,
            truncated,
            info,
        )

    def reset(
            self,
            seed=None,
            options=None
            ) -> Tuple[ndarray | Dict]:
        obs, info = super().reset(seed, options)

        # reset your changes to env here as needed

        return obs, info

    # TODO: Write code here that makes it so the score is printed on the
    # screen in videos. HINT: find a parent method to override.
    