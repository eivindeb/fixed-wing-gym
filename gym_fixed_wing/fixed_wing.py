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

        self.viewer = None

        self.np_random = np.random.RandomState()
        self.obs_norm_mean_mask = []

        obs_low = []
        obs_high = []
        for obs_var in self.cfg["observation"]["states"]:
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
            space_high = action_var.pop("high", None)

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

            space_low = action_var.pop("low", None)

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
            self.action_outside_bounds_cost = self.cfg["action"].get("bounds_outside_cost", 0)
            self.action_bounds_max = np.full(self.action_space.shape, self.cfg["action"].get("scale_high", 1)) *\
                                     self.cfg["action"]["bounds_multiplier"]
            self.action_bounds_min = np.full(self.action_space.shape, self.cfg["action"].get("scale_low", -1)) *\
                                     self.cfg["action"]["bounds_multiplier"]

        self.goal_enabled = self.cfg["target"]["success_streak_req"] > 0

        self.target = None
        self._target_props = None
        self._target_props_init = None

        self.training = False
        self.render_on_reset = False
        self.render_on_reset_kw = {}
        self.save_on_reset = False

        self.sampler = sampler

        self._schedule_level = None
        self.set_schedule_level(0)

    def seed(self, seed=None):
        """
        Seed the random number generator of the flight simulator

        :param seed: (int) seed for random state
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.simulator.seed(seed)
        return [seed]

    def set_schedule_level(self, level):
        """
        Set the schedule level of the environment, e.g. for starting with simple tasks and progressing to more difficult
        scenarios as the agent becomes increasingly proficient.

        :param level: (int) the schedule level
        """
        self._schedule_level = level
        if "states" in self.cfg:
            for state in self.cfg["states"]:
                state = copy.copy(state)
                state_name = state.pop("name")
                convert_to_radians = state.pop("convert_to_radians", False)
                for prop, values in state.items():
                    if isinstance(values, list):
                        val = values[self._schedule_level]
                    else:
                        val = values
                    if convert_to_radians and val is not None:
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
                        if isinstance(v, list):
                            self._target_props_init["states"][state_name][k] = v[self._schedule_level]
                        else:
                            self._target_props_init["states"][state_name][k] = v
            else:
                if isinstance(val, list):
                    self._target_props_init[attr] = val[self._schedule_level]
                else:
                    self._target_props_init[attr] = val

        # TODO: add support for setting state ranges (how to sync across threads, and make sure states are added only once?
        # TODO: this will update state_range twice when loaded on higher level, and therefore do calculations twice unneccesarily
        if self.sampler is not None:
            for state, attrs in self._target_props_init["states"].items():
                if attrs.get("convert_to_radians", False):
                    low, high = np.radians(attrs["low"]), np.radians(attrs["high"])
                else:
                    low, high = attrs["low"], attrs["high"]
                self.sampler.add_state("{}_target".format(state), state_range=(low, high))

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
            self.save_history("test_hist_rl_eval.npy", ["roll", "pitch", "yaw", "Va"], save_targets=True)
            self.save_on_reset = False
        self.steps_count = 0
        self.simulator.reset(state)
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

        return obs

    def step(self, action):
        """
        Perform one step from action chosen by agent.

        :param action: ([float]) the action chosen by the agent
        :return: ([float], float, bool, dict) observation vector, reward, done, extra information about episode on done
        """

        self.history["action"].append(action)

        actuators = [a["name"] for a in self.cfg["action"]["states"]]

        if self.action_outside_bounds_cost > 0:
            action_rew_high = np.where(action > self.action_bounds_max, action - self.action_bounds_max, 0) *\
                self.action_outside_bounds_cost
            action_rew_low = np.where(action < self.action_bounds_min, action - self.action_bounds_min, 0) *\
                self.action_outside_bounds_cost
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

        reward = self.get_reward()
        info = {}
        done = False

        if self.steps_count >= self.steps_max > 0:
            done = True
            info["termination"] = "steps"

        if step_success:
            if self.goal_enabled:
                for state, status in self._get_goal_status().items():
                    self.history["goal"][state].append(status)

                streak_req = self.cfg["target"]["success_streak_req"]
                if self.cfg["target"]["on_success"] != "none" and self._steps_for_current_target >= streak_req and np.mean(self.history["goal"]["all"][-streak_req:]) >= self.cfg["target"]["success_streak_fraction"]:
                    if self.cfg["target"]["success_reward"] == "timesteps":
                        goal_reward = self.steps_max - self.steps_count
                    else:
                        goal_reward = self.cfg["target"]["success_reward"]

                    reward += goal_reward

                    if self.cfg["target"]["on_success"] == "done":
                        done = True
                        info["termination"] = "success"
                    elif self.cfg["target"]["on_success"] == "new":
                        self.sample_target()
                    else:
                        raise Exception("Unexpected goal action {}".format(self.cfg["target"]["action"]))

            resample_every = self.cfg["target"].get("resample_every", 0)
            if resample_every and self._steps_for_current_target >= resample_every:
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
            info["termination"] = step_info["termination"]
            obs = self.get_observation()

        if self.action_outside_bounds_cost > 0:
            reward -= np.sum(np.abs(action_rew_high)) + np.sum(np.abs(action_rew_low))

        if done:
            info["avg_errors"] = {k: np.abs(np.mean(v) / v[0]) if v[0] != 0 else np.nan for k, v in
                                  self.history["error"].items()}
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

            if self.sampler is not None:
                for state in self.history["target"]:
                    self.sampler.add_data_point("{}_target".format(state),
                                                self.history["target"][state][0],
                                                info["success"][state])


        return obs, reward, done, info

    def linear_action_scaling(self, a, direction="forward"):
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
        for target_var_name, props in self._target_props_init["states"].items():
            var_props = {"class": props.get("class", "constant")}

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

            self.target[target_var_name] = initial_value

            self._target_props[target_var_name] = var_props

    def render(self, mode="plot", close=True, block=False, save_path=None):
        """
        Visualize environment history. Plots of action and reward can be enabled through configuration file.

        :param mode: (str) render mode, one of plot for graph representation and animation for 3D animation with blender
        :param close: (bool) if figure should be closed after showing, or reused for next render call
        :param block: (bool) block argument to matplotlib blocking script from continuing until figure is closed
        :param save_path (str) if given, render is saved to this path.
        """
        if self.training and not self.render_on_reset:
            self.render_on_reset = True
            self.render_on_reset_kw = {"mode": mode, "close": close, "save_path": save_path}
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

            plt.show(block=block)

            if close:
                self.viewer = None

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
        self.simulator.save_history(path, states)
        if save_targets:
            res = np.load(path).item()
            for state in self.target.keys():
                if state in res:
                    res[state + "_target"] = self.history["target"][state]
            np.save(path, res)

    def get_reward(self):
        """
        Get the reward for the current state of the environment.

        :return: (float) reward
        """
        reward = 0
        terms = {term["function_class"]: {"val": 0, "weight": term["weight"]} for term in self.cfg["reward"]["terms"]}

        for component in self.cfg["reward"]["states"]:
            if component["state"] == "action":
                if component["type"] == "value":
                    val = np.sum(np.abs(self.history["action"][-1]))
                elif component["type"] == "delta":
                    if self.steps_count > 1:
                        """
                        vals = self.history[component["state"]]
                        cur = vals[-1]
                        past_avg = np.mean(vals[-(component["window_size"] + 1):-1], axis=0)
                        val = np.sum(np.abs(cur - past_avg))
                        """
                        vals = self.history[component["state"]][-component["window_size"]:]
                        distance = np.diff(vals, axis=0)
                        val = np.sum(np.abs(distance))
                    else:
                        val = 0
            elif component["type"] == "value":
                val = self.simulator.state[component["state"]].value
            elif component["type"] == "error":
                val = self._get_error(component["state"])
            elif component["type"] == "int_error":
                val = np.sum(self.history["error"][component["state"]])
            else:
                raise ValueError("Unexpected reward component type {}".format(component["type"]))

            if component["function_class"] == "linear":
                terms[component["function_class"]]["val"] -= np.clip(np.abs(val) / component["scaling"],
                                                                     0,
                                                                     component.get("max", None))
            elif component["function_class"] == "exponential":
                terms[component["function_class"]]["val"] -= val ** 2 / component["scaling"]
            else:
                raise ValueError("Unexpected function class {} for {}".format(component["function_class"],
                                                                              component["state"]))

        for term_class, term_info in terms.items():
            if term_class == "exponential":
                val = -1 + np.exp(term_info["val"])
            elif term_class == "linear":
                val = term_info["val"]
            else:
                raise ValueError("Unexpected function class {}".format(term_class))
            reward += term_info["weight"] * val

        return reward

    def get_observation(self):
        """
        Get the observation vector for current state of the environment.

        :return: ([float]) observation vector
        """
        obs = []
        action_index = {}
        step = self.cfg["observation"].get("step", 1)

        for i in range(1, (self.cfg["observation"]["length"] + (1 if step == 1 else 0)) * step, step):
            obs_i = []
            if i > self.steps_count:
                i = self.steps_count + 1
            for obs_var in self.cfg["observation"]["states"]:
                if obs_var["type"] == "state":
                    obs_i.append(self.simulator.state[obs_var["name"]].history[-i])
                elif obs_var["type"] == "target":
                    if obs_var["value"] == "relative":
                        obs_i.append(self._get_error(obs_var["name"]) if i == 1 else self.history["error"][obs_var["name"]][-i])
                    elif obs_var["value"] == "absolute":
                        obs_i.append(self.target[obs_var["name"]] if i == 1 else self.history["target"][obs_var["name"]][-i])
                    elif obs_var["value"] == "integrator":
                        if self.history is None:
                            obs_i.append(self._get_error(obs_var["name"]))
                        else:
                            obs_i.append(np.sum(self.history["error"][obs_var["name"]][:-i]))
                    else:
                        raise ValueError("Unexpected observation variable target value type: {}".format(obs_var["value"]))
                elif obs_var["type"] == "action":
                    window_size = obs_var.get("window_size", 1)
                    if self.steps_count - i < 0:
                        if self.scale_actions:
                            obs_i.append(0)
                        else:
                            obs_i.append(self.simulator.state[obs_var["name"]].value)
                    else:
                        low_idx, high_idx = -window_size - i + 1, None if i == 1 else -(i - 1)
                        if self.scale_actions:
                            try:
                                a_i = action_index[obs_var["name"]]
                            except:
                                a_i = [action_var["name"] for action_var in self.cfg["action"]["states"]].index(obs_var["name"])
                                action_index[obs_var["name"]] = a_i
                            obs_i.append(np.sum(np.abs(np.diff([a[a_i] for a in self.history["action"][low_idx:high_idx]])), dtype=np.float32))
                        else:
                            obs_i.append(np.sum(np.abs(np.diff(self.simulator.state[obs_var["name"]].history["command"][low_idx:high_idx])), dtype=np.float32))

                else:
                    raise Exception("Unexpected observation variable type: {}".format(obs_var["type"]))
            if self.cfg["observation"]["shape"] == "vector":
                obs.extend(obs_i)
            elif self.cfg["observation"]["shape"] == "matrix":
                obs.append(obs_i)
            else:
                raise ValueError("Unexpected observation shape {}".format(self.cfg["observation"]["shape"]))

        return obs

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
        for state, props in self._target_props_init["states"].items():
            bound = props.get("bound", None)
            if bound is not None:
                err = self._get_error(state)
                if props.get("convert_to_radians", False):
                    bound = np.radians(bound)
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
            else:
                raise ValueError

            if getattr(self.simulator.state[state], "wrap", False) and np.abs(res[state]) > np.pi:
                res[state] = np.sign(res[state]) * (np.abs(res[state]) % np.pi - np.pi)

        return res

if __name__ == "__main__":
    from pyfly.pid_controller import PIDController

    env = FixedWingAircraft("fixed_wing_config.json", config_kw={"steps_max": 500})
    env.seed(0)

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

