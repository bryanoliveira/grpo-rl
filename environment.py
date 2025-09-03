import gymnasium as gym
import numpy as np

class Float32ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(Float32ObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low,
            high=self.observation_space.high,
            shape=self.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, observation):
        return observation.astype(np.float32)


def make(env_id, idx, capture_video, run_name, env_configs, gamma):
    """
    Make an environment.
    Image observations are not supported.
    """
    def thunk():
        configs = env_configs.copy()

        if capture_video and idx == 0:
            configs["render_mode"] = "rgb_array"
            env = gym.make(env_id, **configs)
            env = gym.wrappers.RecordVideo(env, f"runs/{run_name}/videos")
        else:
            env = gym.make(env_id, **configs)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = Float32ObservationWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        if idx == 0:
            print(env)
            print("Observation Space", env.observation_space)
            print("Action Space", env.action_space)
        return env

    return thunk