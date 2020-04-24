import gym
import numpy as np

from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecCheckNan, DummyVecEnv, VecNormalize
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2
from gym_fixed_wing.fixed_wing import FixedWingAircraft
import tensorflow as tf
import time
import os
import io
from PIL import Image
import shutil
import matplotlib.pyplot as plt
from evaluate_controller import evaluate_model_on_set


def save_model(model, save_folder):
    """
    Helper function to save model checkpoint.
    :param model: (PPO2 object) the model to save.
    :param save_folder: (str) the folder to save model to.
    :return:
    """
    model.save(os.path.join(save_folder, "model.pkl"))
    model.env.save_running_average(save_folder)


def monitor_training(_locals, _globals):
    """
    Training callback monitoring progress, saving checkpoints, logging metrics and render figures, and running test set evaluations.
    :param _locals: (dict) local variables of the training loop
    :param _globals: (dict) global variables of the training loop
    :return:
    """
    global curriculum_level, last_ep_info, info_kw, log_interval, curriculum_cooldown, render_interval, last_render, \
        render_check, model_folder, test_interval, last_test, checkpoint_save_interval, last_save, test_set_path, \
        config_path
    if "ep_info_buf" in _locals:
        ep_info_buf = _locals["ep_info_buf"]
    else:
        ep_info_buf = _locals["self"].ep_info_buf
    if len(ep_info_buf) > 0 and ep_info_buf[-1] != last_ep_info:
        last_ep_info = ep_info_buf[-1]

        now = time.time()

        info = {}
        for ep_info in ep_info_buf:
            for k in ep_info.keys():
                if k in info_kw:
                    if k not in info:
                        info[k] = {}
                    if isinstance(ep_info[k], dict):
                        for state, v in ep_info[k].items():
                            if state in info[k]:
                                info[k][state].append(v)
                            else:
                                info[k][state] = [v]
                    else:
                        if "all" in info[k]:
                            info[k]["all"].append(ep_info[k])
                        else:
                            info[k]["all"] = [ep_info[k]]

        if _locals["writer"] is not None:
            if "success" in info:
                summaries = []
                for measure in info_kw:
                    for k, v in info[measure].items():
                        summaries.append(tf.Summary.Value(tag="ep_info/{}_{}".format(measure, k), simple_value=np.nanmean(v)))
                _locals["writer"].add_summary(tf.Summary(value=summaries), _locals["self"].num_timesteps)

        elif _locals["update"] % log_interval == 0 and _locals["update"] != 0:
            for info_k, info_v in info.items():
                print("\n{}:\n\t".format(info_k) + "\n\t".join(["{:<10s}{:.2f}".format(k, np.nanmean(v)) for k, v in info_v.items()]))

        if curriculum_level < 1:
            if curriculum_cooldown <= 0:
                if np.mean(info["success"]["all"]) > curriculum_level:
                    curriculum_level = min(np.mean(info["success"]["all"]) * 2, 1)
                    env.env_method("set_curriculum_level", curriculum_level)
                    curriculum_cooldown = 15
            else:
                curriculum_cooldown -= 1

        if now - last_render >= render_interval:
            env.env_method("render", indices=0, mode="plot", show=False, close=True, save_path=os.path.join(model_folder, "render", str(_locals["self"].num_timesteps)))
            last_render = time.time()

        if test_set_path is not None and _locals["self"].num_timesteps - last_test >= test_interval:
            last_test = _locals["self"].num_timesteps
            evaluate_model_on_set(test_set_path, model, config_path=config_path, num_envs=_locals["self"].env.num_envs,
                                  writer=_locals["writer"], timestep=_locals["self"].num_timesteps)

        if now - render_check["time"] >= 30:
            for render_file in os.listdir(os.path.join(model_folder, "render")):
                if render_file not in render_check["files"]:
                    render_check["files"].append(render_file)
                    summary = tf.Summary.Value(tag="render", image=object_to_summary_image(
                        os.path.join(*[model_folder, "render", render_file])))
                    _locals["writer"].add_summary(tf.Summary(value=[summary]), int(os.path.splitext(render_file)[0]))

        if now - last_save >= checkpoint_save_interval:
            save_model(_locals["self"], model_folder)
            last_save = now

    return True


def object_to_summary_image(object):
    """
    Helper function to convert an image object or matplotlib figure to a tensorboard summary image.
    :param object: (matplotlib figure) The object to be converted
    :return: tensorboard summary Image
    """
    buf = io.BytesIO()
    if isinstance(object, str):
        img = Image.open(object)
        height, width = img.size
        channels = len(img.getbands())
        img.save(buf, format="PNG")
    else:
        height, width = object.get_size_inches() * object.dpi
        channels = 4
        object.savefig(buf, format='png')
        # Closing the object prevents it from being displayed directly inside
        # the notebook.
        plt.close(object)
    buf.seek(0)
    image_string = buf.getvalue()
    buf.close()
    return tf.Summary.Image(height=int(height),
                            width=int(width),
                            colorspace=channels,
                            encoded_image_string=image_string)


def make_env(config_path, rank, seed=0, info_kw=None, sim_config_kw=None):
    """
    Utility function for multiprocessed env.

    :param config_path: (str) path to gym environment configuration file
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    :param info_kw: ([str]) list of entries in info dictionary that the Monitor should extract
    :param sim_config_kw (dict) dictionary of key value pairs to override settings in the configuration file of PyFly
    """
    def _init():
        env = FixedWingAircraft(config_path, sim_config_kw=sim_config_kw)
        env = Monitor(env, filename=None, allow_early_resets=True, info_keywords=info_kw)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


if __name__ == '__main__':
    from argparse import ArgumentParser
    curriculum_level = 0.25  # Initial difficulty level of environment
    curriculum_cooldown = 25  # Minimum number of episodes between environment difficulty adjustments
    render_interval = 600  # Time in seconds between rendering of training episodes
    last_test = 0
    last_render = time.time()
    checkpoint_save_interval = 300
    last_save = time.time()
    last_ep_info = None
    log_interval = 50
    render_check = {"files": [], "time": time.time()}
    info_kw = ["success", "control_variation", "end_error", "total_error", "success_time_frac"]

    parser = ArgumentParser()
    parser.add_argument("model_name", help="Path to model folder. If already exists, configurations will be loaded from this folder and training will resume from checkpoint.")
    parser.add_argument("num_envs", help="Number of processes for environments.")
    parser.add_argument("--env-config-path", required=False, help="Path to configuration for gym environment")
    parser.add_argument("--train-steps", required=False, help="Number of training time steps")
    parser.add_argument("--policy", required=False, help="Type of policy to use (MLP or CNN)")
    parser.add_argument("--disable-curriculum", dest="disable_curriculum", action="store_true", required=False,
                        help="If this flag is set, curriculum (i.e. gradual increase in sampling region of initial and target conditions based on proficiency) is disabled.")
    parser.add_argument("--test-set-path", required=False, help="Path to test set. If supplied, the model is evaluated on this test set 4 times during training.")

    args = parser.parse_args()
    num_cpu = int(args.num_envs)
    test_set_path = args.test_set_path
    if args.policy is None or args.policy == "MLP":
        policy = MlpPolicy
    elif args.policy == "CNN":
        try:
            from stable_baselines.common.policies import CnnMlpPolicy
            policy = CnnMlpPolicy
        except:
            print("To use the CNN policy described in the paper you need to use the stable-baselines fork at github.com/eivindeb/stable-baselines")
            exit(0)
    else:
        raise ValueError("Invalid value supplied for argument policy (must be either 'MLP' or 'CNN')")

    if args.disable_curriculum:
        curriculum_level = 1

    if args.train_steps:
        training_steps = int(args.train_steps)
    else:
        training_steps = int(5e6)

    test_interval = int(training_steps / 5)  # How often in time steps during training the model is evaluated on the test set

    model_folder = os.path.join("models", args.model_name)
    if os.path.exists(model_folder):
        load = True
    else:
        load = False
        if args.env_config_path is None:
            config_path = "fixed_wing_config.json"
        else:
            config_path = args.env_config_path
        os.makedirs(model_folder)
        os.makedirs(os.path.join(model_folder, "render"))
        shutil.copy2(config_path, os.path.join(model_folder, "fixed_wing_config.json"))
    config_path = os.path.join(model_folder, "fixed_wing_config.json")

    env = VecNormalize(SubprocVecEnv([make_env(config_path, i, info_kw=info_kw) for i in range(num_cpu)]))
    env.env_method("set_curriculum_level", curriculum_level)
    env.set_attr("training", True)

    if load:
        model = PPO2.load(os.path.join(model_folder, "model.pkl"), env=env, verbose=1, tensorboard_log=os.path.join(model_folder, "tb"))
    else:
        model = PPO2(policy, env, verbose=1, tensorboard_log=os.path.join(model_folder, "tb"))
    model.learn(total_timesteps=training_steps, log_interval=log_interval, callback=monitor_training)
    save_model(model, model_folder)

