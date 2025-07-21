# logging

Interesting reading:

Best Practices when training with PPO

https://github.com/EmbersArc/PPO/blob/master/best-practices-ppo.md

Comparing Deterministic and Soft Policy Gradients for Optimizing Gaussian Mixture Actors

https://openreview.net/forum?id=qS9pPu8ODt

Must add an extension to do the ackermann or add a bespoke Ackermann controller

The idea is to force the model to constrain itself to realistic values.

Isaaclab extension development:

https://isaac-sim.github.io/IsaacLab/main/source/overview/developer-guide/development.html

Isaaclab Extension template:

https://github.com/isaac-sim/IsaacLabExtensionTemplate

## Improving Policy

Clipping Mechanism, Generalized Advantage Estimation, Adaptive Learning Rate, Reward Shaping, Reward Normalization/Clipping, Noise-Robust Reward Functions, 


Value network instability, Separate Learning Rates, Increased Training, Gradient Clipping

Batch Normalization


## Clip Actions

The clip actions only help to match the expected values by the controller, adding logic at this point does not seem to have any massive benefit.


[Question] Lift Task with UR10e and Robotiq 2F-140 #2932

https://github.com/isaac-sim/IsaacLab/discussions/2932

In this question they address several optimization and tuning ideas

It can still be improved but hyperparameter tuning will only get so far, the oscillations are common and must be accounted by the hardware. The problem seems to be the actor and samples exploration noise which methods like GMMs do seem to offer major improvements. 

Clips actions to large limits before applying them to the environment #984

https://github.com/isaac-sim/IsaacLab/pull/984

They added the clipping to the rsl_rl however it does not seem to be the magical bullet

Adds action clipping to rsl-rl wrapper

https://github.com/ToxicNS/IsaacLab/commit/753460c02fddac95bf7e490c78ee5da8391c204c

## OOM Error

During training the rsl_rl the following OOM happens:

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