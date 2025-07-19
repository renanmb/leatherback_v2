# logging

During training the rsl_rl the following happens:

```bash
################################################################################
                       Learning iteration 339/500                       

                       Computation: 123142 steps/s (collection: 0.980s, learning 0.084s)
             Mean action noise std: 22.31
          Mean value_function loss: 1.2068
               Mean surrogate loss: 0.0020
                 Mean entropy loss: 9.0455
                       Mean reward: 5.16
               Mean episode length: 301.00
--------------------------------------------------------------------------------
                   Total timesteps: 44564480
                    Iteration time: 1.06s
                      Time elapsed: 00:05:49
                               ETA: 00:02:45

Error executing job with overrides: []
Traceback (most recent call last):
  File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py", line 101, in hydra_main
    func(env_cfg, agent_cfg, *args, **kwargs)
  File "/home/goat/Documents/GitHub/renanmb/leatherback_v2/scripts/rsl_rl/train.py", line 183, in main
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
  File "/home/goat/anaconda3/envs/isaaclab/lib/python3.10/site-packages/rsl_rl/runners/on_policy_runner.py", line 204, in learn
    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
  File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py", line 176, in step
    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
  File "/home/goat/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/gymnasium/wrappers/order_enforcing.py", line 56, in step
    return self.env.step(action)
  File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py", line 384, in step
    self.obs_buf = self._get_observations()
  File "/home/goat/Documents/GitHub/renanmb/leatherback_v2/source/leatherback_v2/leatherback_v2/tasks/direct/leatherback_v2/leatherback_env.py", line 206, in _get_observations
    raise ValueError("Observations cannot be NAN")
ValueError: Observations cannot be NAN

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```