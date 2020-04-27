# gym-fixed-wing

The Fixed-Wing aircraft environment is an [OpenAI Gym](https://github.com/openai/gym) wrapper for the 
[PyFly](https://github.com/eivindeb/pyfly) flight simulator, adding several features on top of the base simulator
such as target states and computation of performance metrics. This environment allows for training of reinforcement learning controllers
for attitude control of fixed-wing aircraft.

## Installation

```shell
git clone https://github.com/eivindeb/fixed-wing-gym
cd fixed-wing-gym
pip install -e .
```

## Getting started

The [examples](gym_fixed_wing/examples) folder contains scripts demonstrating how the gym can be integrated with the [stable-baselines library](https://github.com/hill-a/stable-baselines), and how to
reproduce the results presented in the [paper](https://arxiv.org/abs/1911.05478). 

## Documentation

Each functions behaviour, arguments and return values are documented with docstrings in the source code. The Fixed-Wing-Gym is configured through the configuration json file. The config file 
consists of seven main blocks:

### System settings
The first level of the configuration file accepts the following arguments:
* **steps_max** Int. Sets the time limit of the episode, i.e. the maximum number of simulations steps performed in each episode before it is ended.
* **integration_window** Int. Sets the moving window size for states representing the integral of the state in the moving window.

### Observation
The observation block configures the observations returned by the environment as input to the policy.

* **length** Int. The amount of previous timesteps (including the current one) that should be included in the observation vector. 
With length 1, only values for the current timestep is included.
* **step** Int. With length > 1, step is the step size between the timesteps of previous values in the observation vector.
I.e. with length = 3 and step = 2, values for timestep: t, t-3 and t-5 are included.
* **shape** String. One of ["matrix", "vector"]. With length > 1, the shape controls whether previous timestep values are added as rows or concatenated as one vector.
* **normalize** Boolean. Whether the values in the observation vector should be normalized, with normalization factors as provided for each state.
* **noise** Dict. Adds white noise to the observation vector if present with either mean or variance nonzero.
* **states** List. The states from the PyFly simulator that is included in the observation vector. For each state, the
following arguments are available:
    * **name** String. Required. The name of the state in the PyFly simulator.
    * **low** Float. The observation space minimum of this state. If unspecified, the value is taken from PyFly if available, 
    else it is set to minimum of a float 32 bit number.
    * **high** Float. The observation space maximum of this state.
    * **convert_to_radians** Boolean. If set to true, all (angular) values specified for this state are converted to radians.
    * **type** String. One of ["state", "target", "action"]. This argument controls if the value is collected from the PyFly state
    object for "state" and "action", or from the target attribute of fixed-wing-gym for "target". For "action" the value is also scaled
    according to the specified action scaling.
    * **value** String. One of ["relative", "absolute", "integrator"]. For states with type "target". Value "relative" gives the error of the state (i.e. current value - desired value),
    "absolute" gives the current value of the target state (i.e. not desired value), "integrator" gives the integral of the error over the integration window.
    * **mean** Float. For normalization, gives the mean statistic.
    * **var** Float. For normalization, gives the variance statistic.

### Action
The action block configures how the incoming actions are handled.

* **scale_space** Boolean. Controls whether incoming actions are linearly scaled from the state's action space minimum and maximum to the scale_space minimum and maximum.
* **scale_low** Float. If action space is scaled, the action minimum is linearly scaled from the action space minimum to this value.
* **scale_high** Float. If action space is scaled, the action maximum is linearly scaled from the action space maximum to this value.
* **bounds_multiplier** Float. If "out-of-bounds" reward is specified, this argument sets the bounds as a product of the action space bounds and this value.
* **states** List. The actuation states of the PyFly simulator for which input values are provided.
    * **name** String. The name of the actuation state in the PyFly simulator.
    * **low** Float/String. Either "max" or float value. Sets the action space minimum value. If "max" it sets the value to the minimum value of float 32-bit number.
    * **high** Float/String. Either "max" or float value. Sets the action space maximum value. If "max" it sets the value to the maximum value of float 32-bit number.
 

### Target
The target block configures how target states are selected and objective completion is handled.

* **resample_every** Int. If nonzero, a new target is sampled every resample_every simulation steps.
* **success_streak_req** Int. The window size in simulation steps for which objective completion is considered.
* **success_streak_fraction** Float. The fraction of simulation steps in the success streak window for which the goal must be satisfied for the episode to be counted as a success.
* **on_success** String. One of ["done", "new", "none"]. Decides the environments response to a successful episode. 
If "done", the episode is terminated, if "new", a new target is sampled, if "none" then nothing happens.
* **states** List. The states from the PyFly simulator that have defined desired values, and are involved in determining the success of the episode.
    * **name** String. The name of the state in the PyFly simulator.
    * **low** Float. Sets the minimum desired value for the target state.
    * **high** Float. Sets the maximum desired value for the target state.
    * **bound** Float. Sets the goal bounds around the desired value for the target state. If current value of state is inside this bound, the goal criterion for this state is satisfied.
    * **delta** Float. Sets the maximum difference in initial condition for state and the desired target value of that state. If delta is specified,
    the specified minimum and maximum desired values may be violated, i.e. delta takes precedence over low and high.
    * **class** String. One of ["constant", "linear", "sinusoidal"]. Sets the type of target. If "constant", 
    the target value is held constant throughout the episode (or alternatively until resampled), if "linear", a linear function is sampled
    for the target, if "sinusoidal" a sinusoidal function is sampled for the target state.
    * **slope_low** Float. If class is "linear", slope_low sets the minimum slope of the target line.
    * **slope_high** Float. If class is "linear", slope_high sets the maximum slope of the target line.
    * **amplitude_low** Float. If class is "sinusoidal", amplitude_low sets the minimum amplitude of the target state.
    * **amplitude_high** Float. If class is "sinusoidal", amplitude_high sets the maximum amplitude of the target state.
    * **period_low** Float. If class is "sinusoidal", period_low sets the minimum period of the target state (in simulation steps).
    * **period_high** Float. If class is "sinusoidal", period_high sets the maximum period of the target state (in simulation steps).

### Reward
The reward block defines the reward function of the environment.

* **form** String. One of ["absolute", "potential"]. Sets the type of the reward function. If "absolute", the reward function is calculated as is.
 If "potential" the reward function is calculated as changes in a potential field, i.e. reward_t = reward_t_absolute - reward_t-1_absolute.
* **randomize_scaling** Boolean. If true, the scaling (weight) on each factor of the reward function is randomly varied.
* **step_fail** String/Float. Either a numerical value or "timesteps". Defines the reward (or cost) received when a simulation step fails,
e.g. due to constraint violation. If "timesteps", the reward is equal to -1 * steps left in episode.
* **terms** List. The reward function can be defined as a sum of several terms of different function classes. This
list defines which types of terms should be included and their weight.
    * **function_class** String. One of ["linear", "quadratic", "exponential"], sets the function class of the term.
    * **weight** Float. Sets the weight of the term.
* **factors** List. The factors that goes into the reward function.
    * **name** String. The name of the state if class is "state".
    * **class** String. One of ["state", "action", "success", "step", "goal"]. If "state" the value is collected from the state with "name" in 
    PyFly simulators state object. If "action", the value is collected from the supplied actions. If "success" the value is dependent on episode success.
    If "step", the value is applied every step. If "goal" the value is applied every step that a goal criterion is satisfied.
    * **type** String. One of ["value", "delta", "bound", "error", "int_error", "per_state", "all"]. If class is "state" or "action", "value" takes the value of the
    state or action directly as the reward signal. If class is "action", "delta" yields the total change in action values from one timestep to the next summed over window_size, 
    "bound" gives the total value of how far outside the action_bounds specified that the current timesteps actions are. 
    If class is "state", "error" yields the value of current-state - desired_state, "int_error" gives the integral of the error over the integration window.
    If class is "goal", "all" yields a nonzero value only when all goal criterions are satisfied (i.e. all states are within the bounds for the target states),
    , "per_state" gives value 1 / number of goals individually per goal criterion satisfied.
    * **window_size** Int. For rewards that aggregate over several timesteps, this sets the number of timesteps included.
    * **scaling** Float. The weight of this factor in the reward function.
    * **shaping** Boolean. When form is "potential", this argument controls whether this factors value is part of the potential field (i.e. its value compared to the previous timestep),
    or should be included as its absolute value.
    * **sign** Int. One of [1, -1]. Sets the sign of the factor.
    * **max** Float. Sets the maximum value of this factor in the reward function.

### Simulator
Arguments defined in the simulator block overwrites the equivalent argument in the PyFly simulator. Values specified here
are included in the curriculum level functionality, i.e. their sampling ranges are scaled according to some external measure such as the success rate of the controller.

* **states** List. The states listed here are included in the curriculum, and have their arguments scaled according to:
        
        curriculum_level in [0, 1]                                                       # current environment difficulty level
        midpoint = (sample_range_max + sample_range_min) / 2                             # assumed to be the easiest value
        sample_range_xxx = midpoint + curriculum_level * (sample_range_xxx - midpoint).  # replace xxx with min or max
    * **name** String. Name of state. See PyFly documentation for other state arguments (init, constraint etc.)

### Render
The render block controls what is rendered when calling the render function. Any additional plots
are added as subplots to the PyFly render output.

* **plot_action** Boolean. Plot incoming (unscaled and unclipped) actions.
* **plot_reward** Boolean. Plot the rewards for each timestep.
* **plot_target** Boolean. Plot the desired values of each target state as a dashed line in the corresponding state plot.
* **plot_goal** Boolean. Plot when the goal criterion is satisfied for the target state as a shaded region around the desired value with size bound.

### Metrics
The metric block controls what metrics are calculated at the end of the episode and made available to the reinforcement learning framework in the info dictionary at episode termination.
It consists of a list of dictionaries with a required name and optional arguments, e.g. "low" and "high" limits for rise time.

* **name** String. Name of metric.

The available metrics are:
* **avg_error** The error of current to reference value in the target states averaged over the simulation steps and weighted by the inital error. Is set to np.nan if initial error is less than 0.01.
* **control_variation** The average change in actuator commands per second, where the average is taken over simulation steps and actuators.
* **end_error** The error of current to reference value at the last simulation step.
* **overshoot** The maximum error achieved on the opposing side of the reference value wrt. the initial error, as a fraction of the initial error. Is set to np.nan if the state value never crosses the setpoint value.
* **rise_time** The number of simulation steps it takes to reduce the reference value error from high fraction of initial error to low fraction. High and low thresholds are configurable, with defaults 0.9 and 0.1.
* **settling_time** The number of simulation steps it takes to settle within the goal bounds, i.e. never exiting the bounds from that point on.
* **success** Whether or not the goal is achieved.
* **success_time_frac** The fraction of simulation steps for which the error in the target states is lower than the target bounds.
* **total_error** The error of current to reference value in the target states summed over the simulation steps.

## Citation
If you use this software in your work, please consider citing:

```text
@inproceedings{bohn2019deep,
  title={Deep Reinforcement Learning Attitude Control of Fixed-Wing UAVs Using Proximal Policy optimization},
  author={B{\o}hn, Eivind and Coates, Erlend M and Moe, Signe and Johansen, Tor Arne},
  booktitle={2019 International Conference on Unmanned Aircraft Systems (ICUAS)},
  pages={523--533},
  year={2019},
  organization={IEEE}
}
```



 

