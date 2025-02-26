"""Wrappers for gym environment"""
import os 
import numpy as np
from copy import deepcopy
from typing import List, Callable, Optional, SupportsFloat, Any
import gymnasium as gym
from gymnasium import error, logger
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.error import DependencyNotInstalled
from utils.utils import get_time_str

class RecordBestVideo(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """
    Modified from: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/rendering.py#L155

    Records videos of environment episodes using the environment's render function, saves only new-best videos by default.

    .. py:currentmodule:: gymnasium.utils.save_video

    Usually, you only want to record episodes intermittently, say every hundredth episode or at every thousandth environment step.
    To do this, you can specify ``episode_trigger`` or ``step_trigger``.
    They should be functions returning a boolean that indicates whether a recording should be started at the
    current episode or step, respectively.

    The ``episode_trigger`` should return ``True`` on the episode when recording should start.
    The ``step_trigger`` should return ``True`` on the n-th environment step that the recording should be started, where n sums over all previous episodes.
    If neither :attr:`episode_trigger` nor ``step_trigger`` is passed, a default ``episode_trigger`` will be employed, i.e. :func:`capped_cubic_video_schedule`.
    This function starts a video at every episode that is a power of 3 until 1000 and then every 1000 episodes.
    By default, the recording will be stopped once reset is called.
    However, you can also create recordings of fixed length (possibly spanning several episodes)
    by passing a strictly positive value for ``video_length``.

    No vector version of the wrapper exists.

    Examples - Run the environment for 50 episodes, and save the video every 10 episodes starting from the 0th:
        >>> import os
        >>> import gymnasium as gym
        >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
        >>> trigger = lambda t: t % 10 == 0
        >>> env = RecordVideo(env, video_folder="./save_videos1", episode_trigger=trigger, disable_logger=True)
        >>> for i in range(50):
        ...     termination, truncation = False, False
        ...     _ = env.reset(seed=123)
        ...     while not (termination or truncation):
        ...         obs, rew, termination, truncation, info = env.step(env.action_space.sample())
        ...
        >>> env.close()
        >>> len(os.listdir("./save_videos1"))
        5

    Examples - Run the environment for 5 episodes, start a recording every 200th step, making sure each video is 100 frames long:
        >>> import os
        >>> import gymnasium as gym
        >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
        >>> trigger = lambda t: t % 200 == 0
        >>> env = RecordVideo(env, video_folder="./save_videos2", step_trigger=trigger, video_length=100, disable_logger=True)
        >>> for i in range(5):
        ...     termination, truncation = False, False
        ...     _ = env.reset(seed=123)
        ...     _ = env.action_space.seed(123)
        ...     while not (termination or truncation):
        ...         obs, rew, termination, truncation, info = env.step(env.action_space.sample())
        ...
        >>> env.close()
        >>> len(os.listdir("./save_videos2"))
        2

    Examples - Run 3 episodes, record everything, but in chunks of 1000 frames:
        >>> import os
        >>> import gymnasium as gym
        >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
        >>> env = RecordVideo(env, video_folder="./save_videos3", video_length=1000, disable_logger=True)
        >>> for i in range(3):
        ...     termination, truncation = False, False
        ...     _ = env.reset(seed=123)
        ...     while not (termination or truncation):
        ...         obs, rew, termination, truncation, info = env.step(env.action_space.sample())
        ...
        >>> env.close()
        >>> len(os.listdir("./save_videos3"))
        2

    Change logs:
     * v0.25.0 - Initially added to replace ``wrappers.monitoring.VideoRecorder``
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        video_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
        record_mode: str = "best",
        reward_in_name: bool = True,
        second_metric: Optional[str] = None,
    ):
        """Wrapper records videos of rollouts.

        Args:
            env: The environment that will be wrapped
            video_folder (str): The folder where the recordings will be stored
            episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
            step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
            video_length (int): The length of recorded episodes. If 0, entire episodes are recorded.
                Otherwise, snippets of the specified length are captured
            name_prefix (str): Will be prepended to the filename of the recordings
            fps (int): The frame per second in the video. Provides a custom video fps for environment, if ``None`` then
                the environment metadata ``render_fps`` key is used if it exists, otherwise a default value of 30 is used.
            disable_logger (bool): Whether to disable moviepy logger or not, default it is disabled
            
            record_mode: str = "best",
                Mode of when videos are recorded. 'off': never, 'best': only on new best reward, 'all' all videos (like parent wrapper)

            reward_in_name: bool = True,
                If true, adds the reward of the episode to the video file name

            second_metric: Optional[str] = None,
                If a string is provided, looks for a second_metric key in the info dictionary and adds that to the file name as key_value. Only supported on whole episodes
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
        )
        gym.Wrapper.__init__(self, env)

        if env.render_mode in {None, "human", "ansi"}:
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with RecordVideo.",
                "Initialize your environment with a render_mode that returns an image, such as rgb_array.",
            )

        if episode_trigger is None and step_trigger is None:
            from gymnasium.utils.save_video import capped_cubic_video_schedule

            episode_trigger = capped_cubic_video_schedule

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.disable_logger = disable_logger

        self.video_folder = os.path.abspath(video_folder)
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        if fps is None:
            fps = self.metadata.get("render_fps", 30)
        self.frames_per_sec: int = fps
        self.name_prefix: str = name_prefix
        self._video_name: str | None = None
        self.video_length: int = video_length if video_length != 0 else float("inf")
        self.recording: bool = False
        self.recorded_frames: list[RenderFrame] = []
        self.render_history: list[RenderFrame] = []

        self.step_id = -1
        self.episode_id = -1

        try:
            import moviepy  # noqa: F401
        except ImportError as e:
            raise error.DependencyNotInstalled(
                "MoviePy is not installed, run `pip install moviepy`"
            ) from e
        
        assert record_mode in ['best', 'all'], "record mode must be either 'best' or 'all'. Default is 'best'"
        if second_metric: assert step_trigger is None, "second_metric only supports whole episodes"
        self.record_mode = record_mode
        self.reward_in_name = reward_in_name
        self.second_metric = second_metric

        self.best_reward = -np.inf
        self.episode_reward = 0.0
        self.second_metric_value = 0.0

    def _capture_frame(self):
        assert self.recording, "Cannot capture a frame, recording wasn't started."

        frame = self.env.render()
        if isinstance(frame, List):
            if len(frame) == 0:  # render was called
                return
            self.render_history += frame
            frame = frame[-1]

        if isinstance(frame, np.ndarray):
            self.recorded_frames.append(frame)
        else:
            self.stop_recording()
            logger.warn(
                f"Recording stopped: expected type of frame returned by render to be a numpy array, got instead {type(frame)}."
            )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and eventually starts a new recording."""
        obs, info = super().reset(seed=seed, options=options)
        self.episode_id += 1

        if self.recording and self.video_length == float("inf"):
            self.stop_recording()

        if self.episode_trigger and self.episode_trigger(self.episode_id):
            self.start_recording(f"{self.name_prefix}-episode-{self.episode_id}")
        if self.recording:
            self._capture_frame()
            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
        self.episode_reward = 0.0
        self.second_metric_value = 0.0

        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.step_id += 1
        self.episode_reward += rew
        if self.second_metric:
            self.second_metric_value = info[self.second_metric]
            
        if self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")
        if self.recording:
            self._capture_frame()

            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

        return obs, rew, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame]:
        """Compute the render frames as specified by render_mode attribute during initialization of the environment."""
        render_out = super().render()
        if self.recording and isinstance(render_out, List):
            self.recorded_frames += render_out

        if len(self.render_history) > 0:
            tmp_history = self.render_history
            self.render_history = []
            return tmp_history + render_out
        else:
            return render_out

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        if self.recording:
            self.stop_recording()

    def start_recording(self, video_name: str):
        """Start a new recording. If it is already recording, stops the current recording before starting the new one."""
        if self.recording:
            self.stop_recording()

        self.recording = True
        self._video_name = video_name

    def stop_recording(self):
        """Stop current recording and saves the video."""
        assert self.recording, "stop_recording was called, but no recording was started"
        if (self.record_mode == 'best' and (self.episode_reward > self.best_reward)) or self.record_mode == 'all':
            rin = ''
            sm = ''
            if self.reward_in_name:
                rin = f'--reward--{self.episode_reward:.2f}'
            if self.second_metric:
                sm = f'--{self.second_metric}--{self.second_metric_value:.2f}'
            file_name = f'{self._video_name}{rin}{sm}--{get_time_str()}'

            if len(self.recorded_frames) == 0:
                logger.warn("Ignored saving a video as there were zero frames to save.")
            else:
                try:
                    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
                except ImportError as e:
                    raise error.DependencyNotInstalled(
                        "MoviePy is not installed, run `pip install moviepy`"
                    ) from e

                clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
                moviepy_logger = None if self.disable_logger else "bar"
                path = os.path.join(self.video_folder, f"{file_name}.mp4")
                clip.write_videofile(path, logger=moviepy_logger)

        self.recorded_frames = []
        self.recording = False
        self._video_name = None

    def __del__(self):
        """Warn the user in case last video wasn't saved."""
        if len(self.recorded_frames) > 0:
            logger.warn("Unable to save last video! Did you call close()?")
