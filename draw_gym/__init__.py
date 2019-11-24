from gym.envs.registration import register

register(
    id='Drawenv-v0',
    entry_point='draw_gym.draw_env:DrawEnv',
    max_episode_steps=10,
    reward_threshold=0.0,
)