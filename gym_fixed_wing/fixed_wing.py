import gym
from gym.utils import seeding
from pyfly.pyfly import PyFly
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec
import copy
import os
from collections import deque


class FixedWingAircraft(gym.Env):
    def __init__(self, config_path, sampler=None, sim_config_path=None, sim_parameter_path=None, config_kw=None, sim_config_kw=None):
        """
        A gym environment for fixed-wing aircraft, interfacing the python flight simulator PyFly to the openAI
        environment.

        :param config_path: (string) path to json configuration file for gym environment
        :param sim_config_path: (string) path to json configuration file for PyFly
        :param sim_parameter_path: (string) path to aircraft parameter file used by PyFly
        """

        def set_config_attrs(parent, kws):
            for attr, val in kws.items():
                if isinstance(val, dict) or isinstance(parent[attr], list):
                    set_config_attrs(parent[attr], val)
                else:
                    parent[attr] = val

        with open(config_path) as config_file:
            self.cfg = json.load(config_file)

        if config_kw is not None:
            set_config_attrs(self.cfg, config_kw)

        if sim_config_kw is None:
            sim_config_kw = {}
        sim_config_kw.update({"actuation": {"inputs": [a_s["name"] for a_s in self.cfg["action"]["states"]]}})
        pyfly_kw = {"config_kw": sim_config_kw}
        if sim_config_path is not None:
            pyfly_kw["config_path"] = sim_config_path
        if sim_parameter_path is not None:
            pyfly_kw["parameter_path"] = sim_parameter_path
        self.simulator = PyFly(**pyfly_kw)
        self.history = None
        self.steps_count = None
        self.steps_max = self.cfg["steps_max"]
        self._steps_for_current_target = None
        self.goal_achieved = False

        self.integration_window = self.cfg.get("integration_window", 0)

        self.viewer = None

        self.np_random = np.random.RandomState()
        self.obs_norm_mean_mask = []
        self.obs_norm = self.cfg["observation"].get("normalize", False)
        self.obs_module_indices = {"pi": [], "vf": []}

        obs_low = []
        obs_high = []
        for i, obs_var in enumerate(self.cfg["observation"]["states"]):
            self.obs_norm_mean_mask.append(obs_var.get("mask_mean", False))

            high = obs_var.get("high", None)
            if high is None:
                state = self.simulator.state[obs_var["name"]]
                if state.value_max is not None:
                    high = state.value_max
                elif state.constraint_max is not None:
                    high = state.constraint_max
                else:
                    high = np.finfo(np.float32).max
            elif obs_var.get("convert_to_radians", False):
                high = np.radians(high)

            low = obs_var.get("low", None)
            if low is None:
                state = self.simulator.state[obs_var["name"]]
                if state.value_min is not None:
                    low = state.value_min
                elif state.constraint_min is not None:
                    low = state.constraint_min
                else:
                    low = -np.finfo(np.float32).max
            elif obs_var.get("convert_to_radians", False):
                low = np.radians(low)

            if obs_var["type"] == "target" and obs_var["value"] == "relative":
                if high != np.finfo(np.float32).max and low != -np.finfo(np.float32).max:
                    obs_high.append(high-low)
                    obs_low.append(low-high)
                else:
                    obs_high.append(np.finfo(np.float32).max)
                    obs_low.append(-np.finfo(np.float32).max)
            else:
                obs_high.append(high)
                obs_low.append(low)

            if self.obs_norm:
                if obs_var.get("mean", None) is None:
                    if high != np.finfo(np.float32).max and low != -np.finfo(np.float32).max:
                        obs_var["mean"] = high - low
                    else:
                        obs_var["mean"] = 0
                if obs_var.get("var", None) is None:
                    if high != np.finfo(np.float32).max and low != -np.finfo(np.float32).max:
                        obs_var["var"] = (high - low) / (4 ** 2)  # Rule of thumb for variance
                    else:
                        obs_var["var"] = 1

            if obs_var.get("module", "all") != "all":
                self.obs_module_indices[obs_var["module"]].append(i)
            else:
                self.obs_module_indices["pi"].append(i)
                self.obs_module_indices["vf"].append(i)

        self.obs_exclusive_states = True if self.obs_module_indices["pi"] != self.obs_module_indices["vf"] else False

        if self.cfg["observation"]["length"] > 1:
            if self.cfg["observation"]["shape"] == "vector":
                obs_low = obs_low * self.cfg["observation"]["length"]
                obs_high = obs_high * self.cfg["observation"]["length"]
                self.obs_norm_mean_mask = self.obs_norm_mean_mask * self.cfg["observation"]["length"]
            elif self.cfg["observation"]["shape"] == "matrix":
                obs_low = [obs_low for _ in range(self.cfg["observation"]["length"])]
                obs_high = [obs_high for _ in range(self.cfg["observation"]["length"])]
                self.obs_norm_mean_mask = [self.obs_norm_mean_mask for _ in range(self.cfg["observation"]["length"])]
            else:
                raise ValueError

        self.obs_norm_mean_mask = np.array(self.obs_norm_mean_mask)

        action_low = []
        action_space_low = []
        action_high = []
        action_space_high = []
        for action_var in self.cfg["action"]["states"]:
            space_high = action_var.get("high", None)

            state = self.simulator.state[action_var["name"]]
            if state.value_max is not None:
                state_high = state.value_max
            elif state.constraint_max is not None:
                state_high = state.constraint_max
            else:
                state_high = np.finfo(np.float32).max

            if space_high == "max":
                action_space_high.append(np.finfo(np.float32).max)
            elif space_high is None:
                action_space_high.append(state_high)
            else:
                action_space_high.append(space_high)
            action_high.append(state_high)

            space_low = action_var.get("low", None)

            if state.value_min is not None:
                state_low = state.value_min
            elif state.constraint_min is not None:
                state_low = state.constraint_min
            else:
                state_low = -np.finfo(np.float32).max

            if space_low == "max":
                action_space_low.append(-np.finfo(np.float32).max)
            elif space_low is None:
                action_space_low.append(state_low)
            else:
                action_space_low.append(space_low)
            action_low.append(state_low)

        self.observation_space = gym.spaces.Box(low=np.array(obs_low), high=np.array(obs_high), dtype=np.float32)
        self.action_scale_to_low = np.array(action_low)
        self.action_scale_to_high = np.array(action_high)
        # Some agents simply clip produced actions to match action space, not allowing agent to learn that producing
        # actions outside this space is bad.
        self.action_space = gym.spaces.Box(low=np.array(action_space_low),
                                           high=np.array(action_space_high),
                                           dtype=np.float32)

        self.scale_actions = self.cfg["action"].get("scale_space", False)

        if self.cfg["action"].get("bounds_multiplier", None) is not None:
            self.action_bounds_max = np.full(self.action_space.shape, self.cfg["action"].get("scale_high", 1)) *\
                                     self.cfg["action"]["bounds_multiplier"]
            self.action_bounds_min = np.full(self.action_space.shape, self.cfg["action"].get("scale_low", -1)) *\
                                     self.cfg["action"]["bounds_multiplier"]

        self.goal_enabled = self.cfg["target"]["success_streak_req"] > 0

        self.target = None
        self._target_props = None
        self._target_props_init = None
        self._rew_factors_init = copy.deepcopy(self.cfg["reward"]["factors"])

        self.training = False
        self.render_on_reset = False
        self.render_on_reset_kw = {}
        self.save_on_reset = False
        self.save_on_reset_kw = {}

        self.sampler = sampler

        self.prev_shaping = {}

        self._curriculum_level = None
        self.use_curriculum = True
        self.set_curriculum_level(0)

    def seed(self, seed=None):
        """
        Seed the random number generator of the flight simulator

        :param seed: (int) seed for random state
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.simulator.seed(seed)
        return [seed]

    def set_curriculum_level(self, level):  # TODO: implement parameters also?
        """
        Set the curriculum level of the environment, e.g. for starting with simple tasks and progressing to more difficult
        scenarios as the agent becomes increasingly proficient.

        :param level: (int) the curriculum level
        """
        assert 0 <= level <= 1
        self._curriculum_level = level
        if "states" in self.cfg["simulator"]:
            for state in self.cfg["simulator"]["states"]:
                state = copy.copy(state)
                state_name = state.pop("name")
                convert_to_radians = state.pop("convert_to_radians", False)
                for prop, val in state.items():
                    if val is not None:
                        if any([m in prop for m in ["min", "max"]]):
                            midpoint = (state[prop[:-3] + "max"] + state[prop[:-3] + "min"]) / 2
                            val = midpoint - self._curriculum_level * (midpoint - val)
                        if convert_to_radians:
                            val = np.radians(val)
                    setattr(self.simulator.state[state_name], prop, val)

        self._target_props_init = {"states": {}}
        for attr, val in self.cfg["target"].items():
            if attr == "states":
                for state in val:
                    state_name = state.get("name")
                    self._target_props_init["states"][state_name] = {}
                    for k, v in state.items():
                        if k == "name":
                            continue
                        if k not in ["bound", "class"] and v is not None and not isinstance(v, bool):
                            if k == "low":
                                midpoint = (state["high"] + v) / 2
                            elif k == "high":
                                midpoint = (v + state["low"]) / 2
                            else:
                                midpoint = 0

                            v = midpoint - self._curriculum_level * (midpoint - v)
                        self._target_props_init["states"][state_name][k] = v
            else:
                if isinstance(val, list):
                    idx = round(len(val) * self._curriculum_level)
                    self._target_props_init[attr] = val[idx]
                else:
                    self._target_props_init[attr] = val

        if self.sampler is not None:
            for state, attrs in self._target_props_init["states"].items():
                if attrs.get("convert_to_radians", False):
                    low, high = np.radians(attrs["low"]), np.radians(attrs["high"])
                else:
                    low, high = attrs["low"], attrs["high"]
                self.sampler.add_state("{}_target".format(state), state_range=(low, high))

            for state_name in ["roll", "pitch", "velocity_u"]:
                state = self.simulator.state[state_name]
                self.sampler.add_state(state_name, (state.init_min, state.init_max))

        self.use_curriculum = True

    def reset(self, state=None, target=None):
        """
        Reset state of environment.

        :param state: (dict) set initial value of states to given value.
        :param target: (dict) set initial value of targets to given value.
        :return: ([float]) observation vector
        """
        if self.render_on_reset:
            self.render(**self.render_on_reset_kw)
            self.render_on_reset = False
            self.render_on_reset_kw = {}
        if self.save_on_reset:
            self.save_history(**self.save_on_reset_kw)
            self.save_on_reset = False
            self.save_on_reset_kw = {}
        self.steps_count = 0
        if state is None and self.sampler is not None:
            state = {}
            for init_state in ["roll", "pitch", "velocity_u"]:
                state[init_state] = self.sampler.draw_sample(init_state)
        self.simulator.reset(state)
        self.sample_simulator_parameters()
        self.sample_target()
        if target is not None:
            for k, v in target.items():
                if self._target_props[k]["class"] not in ["constant", "compensate"]:
                    self._target_props[k]["class"] = "constant"
                self.target[k] = v

        obs = self.get_observation()
        self.history = {"action": [], "reward": [], "observation": [obs],
                        "target": {k: [v] for k, v in self.target.items()},
                        "error": {k: [self._get_error(k)] for k in self.target.keys()}
                        }
        if self.goal_enabled:
            self.history["goal"] = {}
            for state, status in self._get_goal_status().items():
                self.history["goal"][state] = [status]

        for rew_term in self.cfg["reward"]["terms"]:
            self.prev_shaping[rew_term["function_class"]] = None

        if self.cfg["reward"].get("randomize_scaling", False):
            for i, rew_factor in enumerate(self._rew_factors_init):
                if isinstance(rew_factor["scaling"], list):
                    low, high = rew_factor["scaling"]
                    self.cfg["reward"]["factors"][i]["scaling"] = self.np_random.uniform(low, high)

        return obs

    def step(self, action):
        """
        Perform one step from action chosen by agent.

        :param action: ([float]) the action chosen by the agent
        :return: ([float], float, bool, dict) observation vector, reward, done, extra information about episode on done
        """
        self.history["action"].append(action)

        assert not np.any(np.isnan(action))

        if self.scale_actions:
            action = self.linear_action_scaling(np.clip(action,
                                                        self.cfg["action"].get("scale_low"),
                                                        self.cfg["action"].get("scale_high")
                                                        )
                                                )

        control_input = list(action)

        step_success, step_info = self.simulator.step(control_input)

        self.steps_count += 1
        self._steps_for_current_target += 1

        info = {}
        done = False

        if self.steps_count >= self.steps_max > 0:
            done = True
            info["termination"] = "steps"

        if step_success:
            resample_target = False
            goal_achieved_on_step = False
            if self.goal_enabled:
                for state, status in self._get_goal_status().items():
                    self.history["goal"][state].append(status)

                streak_req = self.cfg["target"]["success_streak_req"]
                if self._steps_for_current_target >= streak_req \
                        and np.mean(self.history["goal"]["all"][-streak_req:]) >= \
                        self.cfg["target"]["success_streak_fraction"]:
                    goal_achieved_on_step = not self.goal_achieved
                    self.goal_achieved = True
                    if self.cfg["target"]["on_success"] == "done":
                        done = True
                        info["termination"] = "success"
                    elif self.cfg["target"]["on_success"] == "new":
                        resample_target = True
                    elif self.cfg["target"]["on_success"] == "none":
                        pass
                    else:
                        raise ValueError("Unexpected goal action {}".format(self.cfg["target"]["action"]))

            reward = self.get_reward(action=self.history["action"][-1],
                                     success=goal_achieved_on_step,
                                     potential=self.cfg["reward"].get("form", "absolute") == "potential")

            resample_every = self.cfg["target"].get("resample_every", 0)
            if resample_target or (resample_every and self._steps_for_current_target >= resample_every):
                self.sample_target()

            for k, v in self._get_next_target().items():
                self.target[k] = v
                self.history["target"][k].append(v)
                self.history["error"][k].append(self._get_error(k))

            obs = self.get_observation()
            self.history["observation"].append(obs)
            self.history["reward"].append(reward)
        else:
            done = True
            reward_fail = self.cfg["reward"].get("step_fail", 0)
            if reward_fail == "timesteps":
                reward = self.steps_count - self.steps_max
            else:
                reward = reward_fail
            info["termination"] = step_info["termination"]
            obs = self.get_observation()

        if done:
            info["avg_error"] = {k: np.abs(np.mean(v) / v[0]) if np.abs(v[0]) >= 0.1 else np.nan for k, v in
                                  self.history["error"].items()}

            # TODO: should handle multiple targets
            info["total_error"] = {}
            for target_state, errors in self.history["error"].items():
                target_values = np.array(self.history["target"][target_state])
                std_traj = np.array(self._get_standard_trajectory(target_state))
                step_min = min([len(errors), std_traj.shape[0], target_values.shape[0]])
                trajectory_error = np.sum(np.abs(errors[:step_min]))
                std_traj_error = np.sum(np.abs(target_values[:step_min] - std_traj[:step_min]))
                info["total_error"][target_state] = trajectory_error / std_traj_error

            info["end_error"] = {k: np.abs(np.mean(v[-50:])) for k, v in self.history["error"].items()}

            control_commands = np.array([self.simulator.state[actuator["name"]].history["command"] for actuator in self.cfg["action"]["states"]])
            delta_controls = np.diff(control_commands, axis=1)
            info["control_variation"] = np.sum(np.abs(delta_controls)) / (3 * self.simulator.dt * delta_controls.shape[1])
            info["settle_time"] = {}
            info["success"] = {}
            for state, goal_status in self.history["goal"].items():
                streak = deque(maxlen=self.cfg["target"]["success_streak_req"])
                settle_time = np.nan
                success = False
                for i, step_goal in enumerate(goal_status):
                    streak.append(step_goal)
                    if len(streak) == self.cfg["target"]["success_streak_req"] and np.mean(streak) >= \
                            self.cfg["target"]["success_streak_fraction"]:
                        settle_time = i
                        success = True
                        break
                info["settle_time"][state] = settle_time
                info["success"][state] = success

            info["success_time_frac"] = {k: np.mean(v) for k, v in self.history["goal"].items()}

            if self.sampler is not None:
                for state in self.history["target"]:
                    self.sampler.add_data_point("{}_target".format(state),
                                                self.history["target"][state][0],
                                                info["success"][state])

                for state in ["roll", "pitch", "velocity_u"]:
                    if state == "velocity_u":
                        self.sampler.add_data_point(state, self.simulator.state["Va"].history[0], info["success"]["Va"])
                    else:
                        self.sampler.add_data_point(state, self.simulator.state[state].history[0], info["success"][state])

        return obs, reward, done, info

    def linear_action_scaling(self, a, direction="forward"):
        """
        Scale input linearly from config parameters scale_high and scale_low to maximum and minimum values of actuators
        reported by PyFly when direction is forward, or vice versa if direction is backward.
        :param a: (np.array of float) action to scale
        :param direction: (str) order of old and new minimums and maximums
        :return: (np.array of float) scaled action
        """
        if direction == "forward":
            new_max = self.action_scale_to_high
            new_min = self.action_scale_to_low
            old_max = self.cfg["action"].get("scale_high")
            old_min = self.cfg["action"].get("scale_low")
        elif direction == "backward":
            old_max = self.action_scale_to_high
            old_min = self.action_scale_to_low
            new_max = self.cfg["action"].get("scale_high")
            new_min = self.cfg["action"].get("scale_low")
        else:
            raise ValueError("Invalid value for direction {}".format(direction))
        return np.array(new_max - new_min) * (a - old_min) / (old_max - old_min) + new_min

    def sample_target(self):
        """
        Set new random target.
        """
        self._steps_for_current_target = 0

        self.target = {}
        self._target_props = {}

        ang_targets = False
        for target_var_name, props in self._target_props_init["states"].items():
            var_props = {"class": props.get("class", "constant")}

            if var_props["class"] == "attitude_angular":
                self.target[target_var_name] = 0
                self._target_props[target_var_name] = props
                ang_targets = True
                continue

            delta = props.get("delta", None)
            convert_to_radians = props.get("convert_to_radians", False)
            low, high = props["low"], props["high"]
            if convert_to_radians:
                low, high = np.radians(low), np.radians(high)
                delta = np.radians(delta) if delta is not None else None
            if delta is not None:
                var_val = self.simulator.state[target_var_name].value
                low = max(low, var_val - delta)
                high = min(high, var_val + delta)

            if self.sampler is None:
                initial_value = self.np_random.uniform(low, high)
            else:
                initial_value = self.sampler.draw_sample("{}_target".format(target_var_name), (low, high))
            if var_props["class"] in "linear":
                var_props["slope"] = self.np_random.uniform(props["slope_low"], props["slope_high"])
                if self.np_random.uniform() < 0.5:
                    var_props["slope"] *= -1
                if convert_to_radians:
                    var_props["slope"] = np.radians(var_props["slope"])
                var_props["intercept"] = initial_value
            elif var_props["class"] == "sinusoidal":
                var_props["amplitude"] = self.np_random.uniform(props["amplitude_low"], props["amplitude_high"])
                if convert_to_radians:
                    var_props["amplitude"] = np.radians(var_props["amplitude"])
                var_props["period"] = self.np_random.uniform(props.get("period_low", 250), props.get("period_high", 500))
                var_props["phase"] = self.np_random.uniform(0, 2 * np.pi) / (2 * np.pi / var_props["period"])
                var_props["bias"] = initial_value - var_props["amplitude"] * np.sin(2 * np.pi / var_props["period"] * (self.steps_count + var_props["phase"]))

            bound = props.get("bound", None)
            if bound is not None:
                var_props["bound"] = bound if not convert_to_radians else np.radians(bound)

            self.target[target_var_name] = initial_value

            self._target_props[target_var_name] = var_props

        if ang_targets:
            assert "roll" in self.target and "pitch" in self.target
            ang_rates = ["omega_q", "omega_r", "omega_p"]
            self.target.update({state: self._attitude_to_angular_rates(state) for state in ang_rates})

    def sample_simulator_parameters(self):
        """
        Sample and set variables and parameters (of UAV mathematical model) of simulator, as specified by simulator
        block in config file.
        :return:
        """
        for key, value in self.cfg["simulator"].items():
            if key == "states":
                continue  # PyFly has its own state randomization procedures
            elif key == "model":
                dist_type = value.get("distribution", "gaussian")
                param_value_var = value["var"]
                param_value_clip = value.get("clip", None)
                for param_arguments in value["parameters"]:
                    orig_param_value = param_arguments.get("original", None)
                    if orig_param_value is None:
                        orig_param_value = self.simulator.params[param_arguments["name"]]
                        param_arguments["original"] = orig_param_value
                    if orig_param_value == 0:
                        continue

                    var = param_arguments.get("var", param_value_var)
                    if value["var_type"] == "relative":
                        var *= np.abs(orig_param_value)
                    if dist_type == "gaussian":
                        param_value = self.np_random.normal(loc=orig_param_value, scale=var)
                        clip = param_arguments.get("clip", param_value_clip)
                        if clip is not None:
                            if value["var_type"] == "relative":
                                clip *= orig_param_value
                            param_value = np.clip(param_value, orig_param_value - clip, orig_param_value + clip)
                    elif dist_type == "uniform":
                        param_value = self.np_random.uniform(low=orig_param_value - var, high=orig_param_value + var)
                    else:
                        raise ValueError("Unexpected distribution type {}".format(dist_type))

                    self.simulator.params[param_arguments["name"]] = param_value
            else:
                if "values" in value:
                    probs = value.get("probabilities", None)
                    if probs is not None:
                        probs = np.array(probs)
                    val = self.np_random.choice(value["values"], p=probs)
                else:
                    val = self.np_random.uniform(value["low"], value["high"])
                    if isinstance(value["low"], bool):
                        val = bool(val)
                setattr(self.simulator, key, val)

    def render(self, mode="plot", show=True, close=True, block=False, save_path=None):
        """
        Visualize environment history. Plots of action and reward can be enabled through configuration file.

        :param mode: (str) render mode, one of plot for graph representation and animation for 3D animation with blender
        :param show: (bool) if true, plt.show is called, if false the figure is returned
        :param close: (bool) if figure should be closed after showing, or reused for next render call
        :param block: (bool) block argument to matplotlib blocking script from continuing until figure is closed
        :param save_path (str) if given, render is saved to this path.
        :return: (matplotlib Figure) if show is false in plot mode, the render figure is returned
        """
        # TODO: handle render call on reset env
        if self.training and not self.render_on_reset:
            self.render_on_reset = True
            self.render_on_reset_kw = {"mode": mode, "show": show, "block": block, "close": close, "save_path": save_path}
            return

        if mode == "plot":
            if self.cfg["render"]["plot_target"]:
                targets = {k: {"data": np.array(v)} for k, v in self.history["target"].items()}
                if self.cfg["render"]["plot_goal"]:
                    for state, status in self.history["goal"].items():
                        if state == "all":
                            continue
                        bound = self._target_props[state].get("bound")
                        targets[state]["bound"] = np.where(status, bound, np.nan)

            self.viewer = {"fig": plt.figure(figsize=(9, 16))}

            extra_plots = 0
            subfig_count = len(self.simulator.plots)
            if self.cfg["render"]["plot_action"]:
                extra_plots += 1
            if self.cfg["render"]["plot_reward"]:
                extra_plots += 1

            subfig_count += extra_plots
            self.viewer["gs"] = matplotlib.gridspec.GridSpec(subfig_count, 1)

            if self.cfg["render"]["plot_action"]:
                labels = [a["name"] for a in self.cfg["action"]["states"]]
                x, y = list(range(len(self.history["action"]))), np.array(self.history["action"])
                ax = plt.subplot(self.viewer["gs"][-extra_plots, 0], title="Actions")
                for i in range(y.shape[1]):
                    ax.plot(x, y[:, i], label=labels[i])
                ax.legend()
            if self.cfg["render"]["plot_reward"]:
                x, y = list(range(len(self.history["reward"]))), self.history["reward"]
                ax = plt.subplot(self.viewer["gs"][-1, 0], title="Reward")
                ax.plot(x, y)

            self.simulator.render(close=close, targets=targets, viewer=self.viewer)

            if save_path is not None:
                if not os.path.isdir(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                _, ext = os.path.splitext(save_path)
                if ext != "":
                    plt.savefig(save_path, bbox_inches="tight", format=ext[1:])
                else:
                    plt.savefig(save_path, bbox_inches="tight")

            if show:
                plt.show(block=block)
                if close:
                    plt.close(self.viewer["fig"])
                    self.viewer = None
            else:
                if close:
                    plt.close(self.viewer["fig"])
                    self.viewer = None
                else:
                    return self.viewer["fig"]

        elif mode == "animation":
            raise NotImplementedError
        else:
            raise ValueError("Unexpected value {} for mode".format(mode))

    def save_history(self, path, states, save_targets=True):
        """
        Save environment state history to file.

        :param path: (string) path to save history to
        :param states: (string or [string]) names of states to save
        :param save_targets: (bool) save targets
        """
        if self.training and not self.save_on_reset:
            self.save_on_reset = True
            self.save_on_reset_kw = {"path": path, "states": states, "save_targets": save_targets}
            return
        self.simulator.save_history(path, states)
        if save_targets:
            res = np.load(path, allow_pickle=True).item()
            for state in self.target.keys():
                if state in res:
                    res[state + "_target"] = self.history["target"][state]
            np.save(path, res)

    def get_reward(self, action=None, success=False, potential=False):
        """
        Get the reward for the current state of the environment.

        :return: (float) reward
        """
        reward = 0
        terms = {term["function_class"]: {"val": 0, "weight": term["weight"], "val_shaping": 0} for term in self.cfg["reward"]["terms"]}

        for component in self.cfg["reward"]["factors"]:
            if component["class"] == "action":
                if component["type"] == "value":
                    val = np.sum(np.abs(self.history["action"][-1]))
                elif component["type"] == "delta":
                    if self.steps_count > 1:
                        vals = self.history[component["name"]][-component["window_size"]:]
                        deltas = np.diff(vals, axis=0)
                        val = np.sum(np.abs(deltas))
                    else:
                        val = 0
                elif component["type"] == "bound":
                    if action is not None:
                        action_rew_high = np.where(action > self.action_bounds_max, action - self.action_bounds_max, 0)
                        action_rew_low = np.where(action < self.action_bounds_min, action - self.action_bounds_min, 0)
                        val = np.sum(np.abs(action_rew_high)) + np.sum(np.abs(action_rew_low))
                    else:
                        val = 0
                else:
                    raise ValueError("Unexpected type {} for reward class action".format(component["type"]))
            elif component["class"] == "state":
                if component["type"] == "value":
                    val = self.simulator.state[component["name"]].value
                elif component["type"] == "error":
                    val = self._get_error(component["name"])
                elif component["type"] == "int_error":
                    val = np.sum(self.history["error"][component["name"]][-self.integration_window:])
                    if self.steps_count < self.integration_window:
                        val += (self.integration_window - self.steps_count) * self.history["error"][component["name"]][0]
                else:
                    raise ValueError("Unexpected reward type {} for class state".format(component["type"]))
            elif component["class"] == "success":
                if success:
                    if component["value"] == "timesteps":
                        val = (self.steps_max - self.steps_count)
                    else:
                        val = component["value"]
                else:
                    val = 0
            elif component["class"] == "step":
                val = component["value"]
            elif component["class"] == "goal":
                val = 0
                if component["type"] == "per_state":
                    for target_state, is_achieved in self._get_goal_status().items():
                        if target_state == "all":
                            continue
                        val += component["value"] / len(self.target) if is_achieved else 0
                elif component["type"] == "all":
                    val += component["value"] if self._get_goal_status()["all"] else 0
                else:
                    raise ValueError("Unexpected reward type {} for class goal".format(component["type"]))
            else:
                raise ValueError("Unexpected reward component type {}".format(component["class"]))

            if component["function_class"] == "linear":
                val = np.clip(np.abs(val) / component["scaling"], 0, component.get("max", None))
            elif component["function_class"] == "exponential":
                val = val ** 2 / component["scaling"]
            elif component["function_class"] == "quadratic":
                val = val ** 2 / component["scaling"]
            else:
                raise ValueError("Unexpected function class {} for {}".format(component["function_class"],
                                                                              component["name"]))

            if component.get("shaping", False):
                terms[component["function_class"]]["val_shaping"] += val * np.sign(component.get("sign", -1))
            else:
                terms[component["function_class"]]["val"] += val * np.sign(component.get("sign", -1))

        for term_class, term_info in terms.items():
            if term_class == "exponential":
                if potential:
                    if self.prev_shaping[term_class] is not None:
                        val = -1 + np.exp(term_info["val"] + (term_info["val_shaping"] - self.prev_shaping["exponential"]))
                    else:
                        val = -1 + np.exp(term_info["val"])
                else:
                    val = -1 + np.exp(term_info["val"] + term_info["val_shaping"])
            elif term_class in ["linear", "quadratic"]:
                val = term_info["val"]
                if potential:
                    if self.prev_shaping[term_class] is not None:
                        val += term_info["val_shaping"] - self.prev_shaping[term_class]
                else:
                    val += term_info["val_shaping"]
            else:
                raise ValueError("Unexpected function class {}".format(term_class))
            self.prev_shaping[term_class] = term_info["val_shaping"]
            reward += term_info["weight"] * val

        return reward

    def get_observation(self):
        """
        Get the observation vector for current state of the environment.

        :return: ([float]) observation vector
        """
        obs = []
        if self.scale_actions:
            action_states = [state["name"] for state in self.cfg["action"]["states"]]
            action_indexes = {state["name"]: action_states.index(state["name"]) for state in self.cfg["observation"]["states"] if state["type"] == "action"}
        step = self.cfg["observation"].get("step", 1)
        init_noise = None
        noise = self.cfg["observation"].get("noise", None)

        for i in range(1, (self.cfg["observation"]["length"] + (1 if step == 1 else 0)) * step, step):
            obs_i = []
            if i > self.steps_count:
                i = self.steps_count + 1
                if self.cfg["observation"]["length"] > 1:
                    init_noise = self.np_random.uniform(-1, 1) * self.simulator.dt
            for obs_var in self.cfg["observation"]["states"]:
                if obs_var["type"] == "state":
                    val = self.simulator.state[obs_var["name"]].history[-i]
                elif obs_var["type"] == "target":
                    if obs_var["value"] == "relative":
                        val = self._get_error(obs_var["name"]) if i == 1 else self.history["error"][obs_var["name"]][-i]
                    elif obs_var["value"] == "absolute":
                        val = self.target[obs_var["name"]] if i == 1 else self.history["target"][obs_var["name"]][-i]
                    elif obs_var["value"] == "integrator":
                        if self.history is None:
                            val = self._get_error(obs_var["name"]) * self.integration_window
                        else:
                            val = np.sum(self.history["error"][obs_var["name"]][-self.integration_window - i:-i])
                            if self.steps_count - i < self.integration_window:
                                val += (self.integration_window - (self.steps_count - i)) * self.history["error"][obs_var["name"]][0]
                    else:
                        raise ValueError("Unexpected observation variable target value type: {}".format(obs_var["value"]))
                elif obs_var["type"] == "action":
                    if self.steps_count - i < 0:
                        val = self.simulator.state[obs_var["name"]].value
                        if self.scale_actions:
                            a_i = action_indexes[obs_var["name"]]
                            action = np.zeros(shape=(len(action_indexes)))
                            action[a_i] = val
                            val = self.linear_action_scaling(action, direction="backward")[a_i]
                    else:
                        window_size = obs_var.get("window_size", 1)
                        low_idx, high_idx = -window_size - i + 1, None if i == 1 else -(i - 1)
                        if self.scale_actions:
                            a_i = action_indexes[obs_var["name"]]
                            val = np.sum(np.abs(np.diff([a[a_i] for a in self.history["action"][low_idx:high_idx]])), dtype=np.float32)
                        else:
                            val = np.sum(np.abs(np.diff(self.simulator.state[obs_var["name"]].history["command"][low_idx:high_idx])), dtype=np.float32)
                else:
                    raise Exception("Unexpected observation variable type: {}".format(obs_var["type"]))
                if init_noise is not None:  # TODO: maybe scale with state range?
                    val += init_noise
                if self.obs_norm and obs_var.get("norm", True):
                    val -= obs_var["mean"]
                    val /= obs_var["var"]
                if noise is not None:
                    val += self.np_random.normal(loc=noise["mean"], scale=noise["var"])
                obs_i.append(val)
            if self.cfg["observation"]["shape"] == "vector":
                obs.extend(obs_i)
            elif self.cfg["observation"]["shape"] == "matrix":
                obs.append(obs_i)
            else:
                raise ValueError("Unexpected observation shape {}".format(self.cfg["observation"]["shape"]))

        return np.array(obs)

    def get_initial_state(self):
        res = {"state": {}, "target": {}}
        for state_name, state_var in self.simulator.state.items():
            if state_name == "attitude":
                continue
            if isinstance(state_var.history, dict):
                res["state"][state_name] = state_var.history["value"][0]
            elif state_var.history is None:
                res["state"][state_name] = 0
            else:
                res["state"][state_name] = state_var.history[0]

        res["target"] = {state: history[0] for state, history in self.history["target"].items()}

        return res

    def get_random_initial_states(self, n_states):
        obs, states = [], []
        for i in range(n_states):
            obs.append(env.reset())
            states.append({"state": self.get_initial_state(), "target": self.target})

        return obs, states

    def get_simulator_parameters(self, normalize=True):
        res = []
        for param in self.cfg["simulator"]["model"]["parameters"]:
            val = self.simulator.params[param["name"]]
            if normalize:
                var_type = self.cfg["simulator"]["model"].get("var_type", "relative")
                var = param.get("var", self.cfg["simulator"]["model"]["var"])
                if var_type == "relative":
                    original_value = param.get("original", self.simulator.params[param["name"]])
                    if original_value == 0:
                        continue
                    var *= original_value
                val = (val - param.get("original", 0)) / var
            res.append(val)

        return res

    def _get_error(self, state):
        """
        Get difference between current value of state and target value.

        :param state: (string) name of state
        :return: (float) error
        """
        if getattr(self.simulator.state[state], "wrap", False):
            return self._get_angle_dist(self.target[state], self.simulator.state[state].value)
        else:
            return self.target[state] - self.simulator.state[state].value

    def _get_angle_dist(self, ang1, ang2):
        """
        Get shortest distance between two angles in [-pi, pi].

        :param ang1: (float) first angle
        :param ang2: (float) second angle
        :return: (float) distance between angles
        """
        dist = (ang2 - ang1 + np.pi) % (2 * np.pi) - np.pi
        if dist < -np.pi:
            dist += 2 * np.pi

        return dist

    def _get_goal_status(self):
        """
        Get current status of whether the goal for each target state as specified by configuration is achieved.

        :return: (dict) status for each and all target states
        """
        goal_status = {}
        for state, props in self._target_props.items():
            bound = props.get("bound", None)
            if bound is not None:
                err = self._get_error(state)
                goal_status[state] = np.abs(err) <= bound

        goal_status["all"] = all(goal_status.values())

        return goal_status

    def _get_next_target(self):
        """
        Get target values advanced by one step.

        :return: (dict) next target states
        """
        res = {}
        for state, props in self._target_props.items():
            var_class = props.get("class", "constant")
            if var_class == "constant":
                res[state] = self.target[state]
            elif var_class == "compensate":
                if state == "Va":
                    if self._target_props["pitch"]["class"] in ["constant", "linear"]:
                        pitch_tar = self.target["pitch"]
                    elif self._target_props["pitch"]["class"] == "sinusoidal":
                        pitch_tar = self._target_props["pitch"]["bias"]
                    else:
                        raise ValueError("Invalid combination of state Va target class compensate and pitch target class {}".format(self._target_props["pitch"]["class"]))
                    if pitch_tar <= np.radians(-2.5):  # Compensate the effects of gravity on airspeed
                        va_target = self.target["Va"]
                        va_end = 28.434 - 40.0841 * pitch_tar
                        if va_target <= va_end:
                            slope = 7 * max(0, 1 if va_target < va_end * 0.95 else 1 - va_target / (va_end * 1.5))
                        else:
                            slope = 0
                        res[state] = va_target + (slope * (-self.target["pitch"]) - 0.25) * self.simulator.dt
                    elif pitch_tar >= np.radians(5):
                        # Converged velocity at 85 % throttle
                        va_end = 26.27 - 41.2529 * pitch_tar
                        va_target = self.target["Va"]
                        if va_target > va_end:
                            if self._steps_for_current_target < 750:
                                res[state] = va_target + (va_end - va_target) * 1 / 150
                            else:
                                res[state] = va_end
                        else:
                            res[state] = va_target
                    else:
                        res[state] = self.target[state]
                elif state == "pitch":
                    raise NotImplementedError
                else:
                    raise ValueError("Unsupported state for target class compensate")
            elif var_class == "linear":
                res[state] = self.target[state] + props["slope"] * self.simulator.dt
            elif var_class == "sinusoidal":
                res[state] = props["amplitude"] * np.sin(2 * np.pi / props["period"] * (self.steps_count + props["phase"])) + props["bias"]
            elif var_class == "attitude_angular":
                if state not in ["omega_p", "omega_q", "omega_r"]:
                    raise ValueError("Invalid state for class attitude_angular {}".format(state))
                res[state] = self._attitude_to_angular_rates(state)
            else:
                raise ValueError

            if getattr(self.simulator.state[state], "wrap", False) and np.abs(res[state]) > np.pi:
                res[state] = np.sign(res[state]) * (np.abs(res[state]) % np.pi - np.pi)

        return res

    def _get_standard_trajectory(self, state, initial_value=None, end_value=None, steps=None):
        if initial_value is None:
            initial_value = self.simulator.state[state].history[0]

        if steps is None:
            steps = self.steps_count

        if state == "roll":
            if end_value is None:
                end_value = self.history["target"][state][-1]
            delta = initial_value - end_value
            L, k, x0, c = delta * 1.18, -0.025, 70, end_value

            values = [L / (1 + np.exp(- k * (step - x0))) + c for step in range(steps)]
            offset = np.radians(0.5)
            values = [v + (offset - np.abs(v - end_value) if np.abs(v - end_value) < offset else 0) for v in values]
        elif state == "pitch":
            if end_value is None:
                end_value = self.history["target"][state][-1]
            delta = initial_value - end_value
            delta_pitch_max = 40
            delta_mag = min(1, np.abs(delta) / delta_pitch_max)
            k = -(0.25 - delta_mag * 0.15)
            x0 = 15 + delta_mag * 20
            L, c = delta * 1.02, np.sign(delta) * delta * 0.005 + end_value

            values = [L / (1 + np.exp(- k * (step - x0))) + c for step in range(steps)]

            offset = np.radians(0.5)
            values = [v + (offset - np.abs(v - end_value) if np.abs(v - end_value) < offset else 0) for v in values]
        elif state == "Va":
            if end_value is None:
                end_value = self.history["target"][state]
            values = [initial_value]
            offset = 0.1
            for step in range(steps - 1):
                if isinstance(end_value, list):
                    end_value_i = end_value[step]
                else:
                    end_value_i = end_value
                val = values[step] + (end_value_i - values[step]) * 1 / 75
                if np.abs(val - end_value_i) < offset:
                    val += offset - np.abs(val - end_value_i)
                values.append(val)
        else:
            raise ValueError("Non target state {} in _get_standard_trajectory".format(state))

        return values

    def _attitude_to_angular_rates(self, state):
        max_vel = self._target_props[state].get("max_vel", np.radians(180))

        roll_angle = self.simulator.state["roll"].value
        pitch_angle = self.simulator.state["pitch"].value

        roll_error = self._get_error("roll")
        pitch_error = self._get_error("pitch")

        # TODO: Evenly distribute between q and r or randomly?
        # TODO: stop angular rates from oscillating between positive and negative when roll angle is maximal

        q_weight_pitch = np.cos(roll_angle)
        r_weight_pitch = np.sin(roll_angle)

        max_pitch_change = max_vel * self.simulator.dt * (q_weight_pitch + r_weight_pitch)

        if state == "omega_p":
            if roll_error <= np.radians(0.1):
                damping = 0.05
            #damping = ((np.abs(roll_error / (0.5 * np.pi))) ** 2) / (np.abs(roll_error / (0.5 * np.pi)))
            damping = np.abs(roll_error / (0.5 * np.pi))
            q_roll = np.sin(roll_angle) * np.tan(pitch_angle) * self.target["omega_q"] * self.simulator.dt
            r_roll = np.cos(roll_angle) * np.tan(pitch_angle) * self.target["omega_r"] * self.simulator.dt
            res = np.clip(-(roll_error - q_roll - r_roll) / self.simulator.dt, -max_vel, max_vel)
        elif state == "omega_q":
            if pitch_error <= np.radians(0.1):
                damping = 0.05
            #damping = ((np.abs(pitch_error / (0.5 * np.pi))) ** 2) / (np.abs(pitch_error / (0.5 * np.pi)))
            damping = np.abs(pitch_error / (0.5 * np.pi))
            if max_pitch_change > np.abs(pitch_error):
                res = - pitch_error / (2 * q_weight_pitch)
            else:
                res = np.sign(q_weight_pitch) * max_vel * np.sign(pitch_error)
        elif state == "omega_r":
            if pitch_error <= np.radians(0.1):
                damping = 0.05
            #damping = ((np.abs(pitch_error / (0.5 * np.pi))) ** 2) / (np.abs(pitch_error / (0.5 * np.pi)))
            damping = np.abs(pitch_error / (0.5 * np.pi))
            if max_pitch_change > np.abs(pitch_error):
                res = pitch_error / r_weight_pitch
            else:
                res = - np.sign(r_weight_pitch) * max_vel * np.sign(pitch_error)

        if np.isnan(damping):
            damping = 0.05
        else:
            damping = min(1, damping)

        res = np.clip(self.target[state] + (res * damping - self.target[state]) * 1 / 20, -max_vel, max_vel)

        return res


class FixedWingAircraftGoal(FixedWingAircraft, gym.GoalEnv):
    def __init__(self, config_path, sampler=None, sim_config_path=None, sim_parameter_path=None, config_kw=None, sim_config_kw=None):
        super(FixedWingAircraftGoal, self).__init__(config_path=config_path,
                                                sampler=sampler,
                                                sim_config_path=sim_config_path,
                                                sim_parameter_path=sim_parameter_path,
                                                config_kw=config_kw,
                                                sim_config_kw=sim_config_kw
                                                )

        self.goal_states = [goal["name"] for goal in self.cfg["observation"]["goals"]]
        self.goal_means = np.array([goal["mean"] for goal in self.cfg["observation"]["goals"]])
        self.goal_vars = np.array([goal["var"] for goal in self.cfg["observation"]["goals"]])

        if len(self.observation_space.shape) == 1:
            goal_space_shape = (len(self.goal_states),)
        else:
            goal_space_shape = (self.observation_space.shape[0], len(self.goal_states))
        self.observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=goal_space_shape, dtype="float32"),
            achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=goal_space_shape, dtype="float32"),
            observation=self.observation_space
        ))

        for module in ["pi", "vf"]:
            self.obs_module_indices[module].extend(list(
                range(len(self.cfg["observation"]["states"]),
                len(self.cfg["observation"]["states"]) + 2 * len(self.goal_states))))

    def reset(self, state=None, target=None):
        super(FixedWingAircraftGoal, self).reset()
        return super(FixedWingAircraftGoal, self).reset(state, target)

    def get_observation(self):
        def scale_goal_states(states, m, v):
            states = np.array(states)
            return (states - m) / v
        obs = super(FixedWingAircraftGoal, self).get_observation()

        # TODO: might have to do some smart pattern thing here
        achieved_goal = scale_goal_states([self.simulator.state[goal_state].value for goal_state in self.goal_states], self.goal_means, self.goal_vars)
        desired_goal = scale_goal_states([self.target[state] for state in self.goal_states], self.goal_means, self.goal_vars)
        if self.cfg["observation"]["length"] > 1:
            achieved_goal = np.repeat(achieved_goal[np.newaxis, :], self.cfg["observation"]["length"], axis=0)
            desired_goal = np.repeat(desired_goal[np.newaxis, :], self.cfg["observation"]["length"], axis=0)

        obs = dict(
            desired_goal=desired_goal,
            achieved_goal=achieved_goal,
            observation=obs
        )

        return obs

    def get_goal_limits(self):
        low, high = [], []
        for i, goal_state in enumerate(self.goal_states):
            goal_cfg = [state for state in self.cfg["target"]["states"] if state["name"] == goal_state][0]
            l, h = goal_cfg["low"], goal_cfg["high"]
            if goal_cfg.get("convert_to_radians", False):
                l, h = np.radians(l), np.radians(h)
            l, h = (l - self.goal_means[i]) / self.goal_vars[i], (h - self.goal_means[i]) / self.goal_vars[i]
            low.append(l)
            high.append(h)

        return np.array(low), np.array(high)

    def compute_reward(self, achieved_goal, desired_goal, info):
        original_values = {"achieved": {}, "desired": {}, "action_history": copy.deepcopy(self.history["action"]),
                           "steps_count": self.steps_count}

        achieved_goal = achieved_goal * self.goal_vars + self.goal_means
        desired_goal = desired_goal * self.goal_vars + self.goal_means

        action = info.get("action", np.array(np.zeros(shape=(len(self.cfg["action"]["states"])))))
        self.history["action"] = self.history["action"][:info["step"]] # TODO: this assumes that this function is called with transitions from the trajectory currently saved in the environment (might not work for multiprocessing etc., and if reset)
        self.history["action"].append(action)
        self.steps_count = info["step"]
        success = False  # TODO: dont know if i want to use this, is get_goal_status in any case

        for i, goal_state in enumerate(self.goal_states):
            original_values["achieved"][goal_state] = self.simulator.state[goal_state].value
            original_values["desired"][goal_state] = self.target[goal_state]
            self.target[goal_state] = desired_goal[i]

        potential = self.cfg["reward"]["form"] == "potential"
        if potential:
            original_values["prev_shaping"] = self.prev_shaping
            if info["step"] > 0:
                # Update prev_shaping to state before transition
                prev_action = self.history["action"][-2] if len(self.history["action"]) >= 2 else None
                for i, goal_state in enumerate(self.goal_states):
                    self.simulator.state[goal_state].value = info["prev_state"][i]
                _ = super(FixedWingAircraftGoal, self).get_reward(action=prev_action, success=success, potential=potential)
            else:
                for rew_term in self.cfg["reward"]["terms"]:
                    self.prev_shaping[rew_term["function_class"]] = None

        for i, goal_state in enumerate(self.goal_states):
            self.simulator.state[goal_state].value = achieved_goal[i]

        reward = super(FixedWingAircraftGoal, self).get_reward(action=action, success=success, potential=potential)

        for goal_state in original_values["achieved"]:
            self.simulator.state[goal_state].value = original_values["achieved"][goal_state]
            self.target[goal_state] = original_values["desired"][goal_state]

        self.history["action"] = original_values["action_history"]
        self.steps_count = original_values["steps_count"]
        if potential:
            self.prev_shaping = original_values["prev_shaping"]

        return reward


if __name__ == "__main__":
    from pyfly.pid_controller import PIDController

    env = FixedWingAircraft("fixed_wing_config.json", config_kw={"steps_max": 1000,
                                                                 "observation": {"noise": {"mean": 0, "var": 0}},
                                                                 "action": {"scale_space": False}})
    env.seed(2)
    env.set_curriculum_level(1)
    obs = env.reset()

    pid = PIDController(env.simulator.dt)
    done = False

    while not done:
        pid.set_reference(env.target["roll"], env.target["pitch"], env.target["Va"])
        phi = env.simulator.state["roll"].value
        theta = env.simulator.state["pitch"].value
        Va = env.simulator.state["Va"].value
        omega = env.simulator.get_states_vector(["omega_p", "omega_q", "omega_r"])

        action = pid.get_action(phi, theta, Va, omega)
        obs, rew, done, info = env.step(action)
    env.render(block=True)

