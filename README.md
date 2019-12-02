# draw_gym
Reinforcement Learning environment for drawing

## Install
```bash
git clone https://github.com/mthorrell/draw_gym.git
cd draw_gym
pip install ./
```

## Example Script
To train PPO2 from [stable-baselines](https://github.com/hill-a/stable-baselines):
```python
import draw_gym
import gym

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('Drawenv-v0')
env = DummyVecEnv([lambda: env])

model = PPO2(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
```
Now, render a single drawing:
```
obs = env.reset()
for i in range(9):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

env.render()
```
## Notes
The package in this example, `stable_baselines`, requires an older version of `tensorflow`. `tensorflow==1.13.2` works for this example.

`draw_gym` downloads the MNIST dataset using `pytorch` helper functions.  This should take ~100M of disk space.