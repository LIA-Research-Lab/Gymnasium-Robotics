"""Environment using Gymnasium API and Multi-goal API for kitchen and Franka robot.

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

This code was also implemented over the repository relay-policy-learning on GitHub (https://github.com/google-research/relay-policy-learning),
published in Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning, by
Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, Karol Hausman.

Original Author of the code: Abhishek Gupta & Justin Fu

The modifications made involve separatin the Kitchen environment from the Franka environment and addint support for compatibility with
the Gymnasium and Multi-goal API's.

This project is covered by the Apache 2.0 License.
"""

from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.envs.franka_kitchen.franka_env import FrankaRobot
from gymnasium_robotics.utils.mujoco_utils import get_joint_qpos

#from mujoco import mj_name2id, mjOBJ_BODY
import mujoco

OBS_ELEMENT_GOALS = {
    "bottom_right_burner": np.array([-0.085, 0.6409, 2.22646769]),
    "bottom_left_burner": np.array([-0.208, 0.6409, 2.22646769]),
    "top_right_burner": np.array([-0.085, 0.6409, 2.34046769]),
    "top_left_burner": np.array([-0.208, 0.6409, 2.34046769]),
    "light_switch": np.array([-0.42413573, 0.61301656, 2.28]),
    "slide_cabinet": np.array([0.292, 0.607, 2.6]),
    "left_hinge_cabinet": np.array([-1.0265508, 0.38476558, 2.6]),
    "right_hinge_cabinet": np.array([-0.16461378,  0.3874147, 2.6]),
    "microwave": np.array([-1.05611982, -0.0269469, 1.792]),
    "kettle": np.array([-0.23,  0.75,  1.62]),
}

OBS_ELEMENT_SITES = {
    "bottom_right_burner": 'knob1_test',
    "bottom_left_burner": 'knob2_test',
    "top_right_burner": 'knob3_test',
    "top_left_burner": 'knob4_test',
    "light_switch": 'light_site',
    "slide_cabinet": 'slide_site',
    "left_hinge_cabinet": 'hinge_site1',
    "right_hinge_cabinet": 'hinge_site2',
    "microwave": 'microhandle_site',
    "kettle": 'kettle',
}

BONUS_THRESH = 0.3


class KitchenEnv(GoalEnv, EzPickle):
    """
    ## Description

    This environment was introduced in ["Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning"](https://arxiv.org/abs/1910.11956)
    by Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, Karol Hausman.

    The environment is based on the 9 degrees of freedom [Franka robot](https://www.franka.de/). The Franka robot is placed in a kitchen environment containing several common
    household items: a microwave, a kettle, an overhead light, cabinets, and an oven. The environment is a `multitask` goal in which the robot has to interact with the previously
    mentioned items in order to reach a desired goal configuration. For example, one such state is to have the microwave andv sliding cabinet door open with the kettle on the top burner
    and the overhead light on. The final tasks to be included in the goal can be configured when the environment is initialized.

    Also, some updates have been done to the original MuJoCo model, such as using an inverse kinematic controller and substituting the legacy Franka model with the maintained MuJoCo model
    in [deepmind/mujoco_menagerie](https://github.com/deepmind/mujoco_menagerie).

    ## Goal

    The goal has a multitask configuration. The multiple tasks to be completed in an episode can be set by passing a list of tasks to the argument`tasks_to_complete`. For example, to open
    the microwave door and move the kettle create the environment as follows:

    ```python
    import gymnasium as gym
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'])
    ```

    The following is a table with all the possible tasks and their respective joint goal values:

    | Task (Joint Name in XML) | Description                                                    | Joint Type | Goal                                       |
    | ------------------------ | -------------------------------------------------------------- | ---------- | ------------------------------------------ |
    | "bottom_right_burner"    | Turn the knob oven knob that activates the bottom right burner | slide      | -0.01                                      |
    | "bottom_left_burner"     | Turn the knob oven knob that activates the bottom right burner | slide      | -0.01                                      |
    | "top_right_burner"       | Turn the knob oven knob that activates the bottom right burner | slide      | -0.01                                      |
    | "top_left_burner"        | Turn the knob oven knob that activates the bottom right burner | slide      | -0.01                                      |
    | light_switch"            | Turn on the light switch                                       | hinge      | -0.7                                       |
    | "slide_cabinet"          | Open the slide cabinet                                         | hinge      | 0.37                                       |
    | "left_hinge_cabinet"     | Open the left hinge cabinet                                    | hinge      | -1.45                                      |
    | "right_hinge_cabinet"    | Open the right hinge cabinet                                   | hinge      | 1.45                                       |
    | "microwave"              | Open the microwave door                                        | hinge      | -0.75                                      |
    | "kettle"                 | Move the kettle to the top left burner                         | free       | [-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06] |


    ## Action Space

    The action space of this environment is different from its original implementation in ["Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning"](https://arxiv.org/abs/1910.11956).
    The environment can be initialized with two possible robot controls: an end-effector `inverse kinematic controller` or a `joint position controller`.

    ### IK Controller
    This action space can be used by setting the argument `ik_controller` to `True`. The space is a `Box(-1.0, 1.0, (7,), float32)`, which represents the end-effector's desired pose displacement.
    The end-effector's frame of reference is defined with a MuJoCo site named `EEF`. The controller relies on the model's position control of the actuators by using the Damped Least Squares (DLS)
    algorithm to compute displacements in the joint space from a desired end-effector configuration. The controller iteratively reduces the error from the current end-effector pose with respect
    to the desired target. The default number of controller steps per Gymnasium step is `5`. This value can be set with the `control_steps` argument, however it is not recommended to reduce this value
    since the simulation can get unstable.

    | Num | Action                                                                            | Action Min  | Action Max  | Control Min  | Control Max | Unit        |
    | --- | ----------------------------------------------------------------------------------| ----------- | ----------- | ------------ | ----------  | ----------- |
    | 0   | Linear displacement of the end-effector's current position in the x direction     | -1          | 1           | -0.2 (m)     | 0.2 (m)     | position (m)|
    | 1   | Linear displacement of the end-effector's current position in the y direction     | -1          | 1           | -0.2 (m)     | 0.2 (m)     | position (m)|
    | 2   | Linear displacement of the end-effector's current position in the z direction     | -1          | 1           | -0.2 (m)     | 0.2 (m)     | position (m)|
    | 3   | Angular displacement of the end-effector's current orientation around the x axis  | -1          | 1           | -0.5 (rad)   | 0.5 (rad)   | angle (rad) |
    | 4   | Angular displacement of the end-effector's current orientation around the y axis  | -1          | 1           | -0.5 (rad)   | 0.5 (rad)   | angle (rad) |
    | 5   | Angular displacement of the end-effector's current orientation around the z axis  | -1          | 1           | -0.5 (rad)   | 0.5 (rad)   | angle (rad) |
    | 6   | Force applied to the slide joint of the gripper                                   | -1          | 1           | 0.0 (N)      | 255 (N)     | force (N)   |

    ### Joint Position Controller
    The default control actuators in the Franka MuJoCo model are joint position controllers. The input action is denormalize from `[-1,1]` to the actuator range and input
    directly to the model. The space is a `Box(-1.0, 1.0, (8,), float32)`, and it can be used by setting the argument `ik_controller` to `False`.

    | Num | Action                                              | Action Min  | Action Max  | Control Min  | Control Max | Name (in corresponding XML file) | Joint | Unit        |
    | --- | ----------------------------------------------------| ----------- | ----------- | -----------  |-------------| ---------------------------------| ----- | ----------- |
    | 0   | `robot:joint1` angular position                     | -1          | 1           | -2.9 (rad)   | 2.9 (rad)   | actuator1                        | hinge | angle (rad) |
    | 1   | `robot:joint2` angular position                     | -1          | 1           | -1.76 (rad)  | 1.76 (rad)  | actuator2                        | hinge | angle (rad) |
    | 2   | `robot:joint3` angular position                     | -1          | 1           | -2.9 (rad)   | 2.9 (rad)   | actuator3                        | hinge | angle (rad) |
    | 3   | `robot:joint4` angular position                     | -1          | 1           | -3.07 (rad)  | 0.0 (rad)   | actuator4                        | hinge | angle (rad) |
    | 4   | `robot:joint5` angular position                     | -1          | 1           | -2.9 (rad)   | 2.9 (rad)   | actuator5                        | hinge | angle (rad) |
    | 5   | `robot:joint6` angular position                     | -1          | 1           | -0.0 (rad)   | 3.75 (rad)  | actuator6                        | hinge | angle (rad) |
    | 6   | `robot:joint7` angular position                     | -1          | 1           | -2.9 (rad)   | 2.9 (rad)   | actuator7                        | hinge | angle (rad) |
    | 7   | `robot:finger_left` and `robot:finger_right` force  | -1          | 1           | 0 (N)        | 255 (N)     | actuator8                        | slide | force (N)   |

    ## Observation Space

    The observation is a `goal-aware observation space`. The observation space contains the following keys:

    * `observation`: this is a `Box(-inf, inf, shape=(59,), dtype="float64")` space and it is formed by the robot's joint positions and velocities, as well as
        the pose and velocities of the kitchen items. An additional uniform noise of ranfe `[-1,1]` is added to the observations. The noise is also scaled by a factor
        of `robot_noise_ratio` and `object_noise_ratio` given in the environment arguments. The elements of the `observation` array are the following:


    | Num   | Observation                                           | Min      | Max      | Joint Name (in corresponding XML file)   | Joint Type | Unit                       |
    | ----- | ----------------------------------------------------- | -------- | -------- | ---------------------------------------- | ---------- | -------------------------- |
    | 0     | `robot:joint1` hinge joint angle value                | -Inf     | Inf      | robot:joint1                             | hinge      | angle (rad)                |
    | 1     | `robot:joint2` hinge joint angle value                | -Inf     | Inf      | robot:joint2                             | hinge      | angle (rad)                |
    | 2     | `robot:joint3` hinge joint angle value                | -Inf     | Inf      | robot:joint3                             | hinge      | angle (rad)                |
    | 3     | `robot:joint4` hinge joint angle value                | -Inf     | Inf      | robot:joint4                             | hinge      | angle (rad)                |
    | 4     | `robot:joint5` hinge joint angle value                | -Inf     | Inf      | robot:joint5                             | hinge      | angle (rad)                |
    | 5     | `robot:joint6` hinge joint angle value                | -Inf     | Inf      | robot:joint6                             | hinge      | angle (rad)                |
    | 6     | `robot:joint7` hinge joint angle value                | -Inf     | Inf      | robot:joint7                             | hinge      | angle (rad)                |
    | 7     | `robot:finger_joint1` slide joint translation value   | -Inf     | Inf      | robot:finger_joint1                      | slide      | position (m)               |
    | 8     | `robot:finger_joint2` slide joint translation value   | -Inf     | Inf      | robot:finger_joint2                      | slide      | position (m)               |
    | 9     | `robot:joint1` hinge joint angular velocity           | -Inf     | Inf      | robot:joint1                             | hinge      | angular velocity (rad/s)   |
    | 10    | `robot:joint2` hinge joint angular velocity           | -Inf     | Inf      | robot:joint2                             | hinge      | angular velocity (rad/s)   |
    | 11    | `robot:joint3` hinge joint angular velocity           | -Inf     | Inf      | robot:joint3                             | hinge      | angular velocity (rad/s)   |
    | 12    | `robot:joint4` hinge joint angular velocity           | -Inf     | Inf      | robot:joint4                             | hinge      | angular velocity (rad/s)   |
    | 13    | `robot:joint5` hinge joint angular velocity           | -Inf     | Inf      | robot:joint5                             | hinge      | angular velocity (rad/s)   |
    | 14    | `robot:joint6` hinge joint angular velocity           | -Inf     | Inf      | robot:joint6                             | hinge      | angular velocity (rad/s)   |
    | 15    | `robot:joint7` hinge joint angular velocity           | -Inf     | Inf      | robot:joint7                             | hinge      | angle (rad)                |
    | 16    | `robot:finger_joint1` slide joint linear velocity     | -Inf     | Inf      | robot:finger_joint1                      | slide      | linear velocity (m/s)      |
    | 17    | `robot:finger_joint2` slide joint linear velocity     | -Inf     | Inf      | robot:finger_joint2                      | slide      | linear velocity (m/s)      |
    | 18    | Rotation of the knob for the bottom right burner      | -Inf     | Inf      | knob_Joint_1                             | hinge      | angle (rad)                |
    | 19    | Joint opening of the bottom right burner              | -Inf     | Inf      | bottom_right_burner                      | slide      | position (m)               |
    | 20    | Rotation of the knob for the bottom left burner       | -Inf     | Inf      | knob_Joint_2                             | hinge      | angle (rad)                |
    | 21    | Joint opening of the bottom left burner               | -Inf     | Inf      | bottom_left_burner                       | slide      | position (m)               |
    | 22    | Rotation of the knob for the top right burner         | -Inf     | Inf      | knob_Joint_3                             | hinge      | angle (rad)                |
    | 23    | Joint opening of the top right burner                 | -Inf     | Inf      | top_right_burner                         | slide      | position (m)               |
    | 24    | Rotation of the knob for the top left burner          | -Inf     | Inf      | knob_Joint_4                             | hinge      | angle (rad)                |
    | 25    | Joint opening of the top left burner                  | -Inf     | Inf      | top_left_burner                          | slide      | position (m)               |
    | 26    | Joint angle value of the overhead light switch        | -Inf     | Inf      | light_switch                             | slide      | position (m)               |
    | 27    | Opening of the overhead light joint                   | -Inf     | Inf      | light_joint                              | hinge      | angle (rad)                |
    | 28    | Translation of the slide cabinet joint                | -Inf     | Inf      | slide_cabinet                            | slide      | position (m)               |
    | 29    | Rotation of the joint in the left hinge cabinet       | -Inf     | Inf      | left_hinge_cabinet                       | hinge      | angle (rad)                |
    | 30    | Rotation of the joint in the right hinge cabinet      | -Inf     | Inf      | right_hinge_cabinet                      | hinge      | angle (rad)                |
    | 31    | Rotation of the joint in the microwave door           | -Inf     | Inf      | microwave                                | hinge      | angle (rad)                |
    | 32    | Kettle's x coordinate                                 | -Inf     | Inf      | kettle                                   | free       | position (m)               |
    | 33    | Kettle's y coordinate                                 | -Inf     | Inf      | kettle                                   | free       | position (m)               |
    | 34    | Kettle's z coordinate                                 | -Inf     | Inf      | kettle                                   | free       | position (m)               |
    | 35    | Kettle's x quaternion rotation                        | -Inf     | Inf      | kettle                                   | free       | -                          |
    | 36    | Kettle's y quaternion rotation                        | -Inf     | Inf      | kettle                                   | free       | -                          |
    | 37    | Kettle's z quaternion rotation                        | -Inf     | Inf      | kettle                                   | free       | -                          |
    | 38    | Kettle's w quaternion rotation                        | -Inf     | Inf      | kettle                                   | free       | -                          |
    | 39    | Bottom right burner knob angular velocity             | -Inf     | Inf      | knob_Joint_1                             | hinge      | angular velocity (rad/s)   |
    | 40    | Opening linear velocity  of the bottom right burner   | -Inf     | Inf      | bottom_right_burner                      | slide      | velocity (m/s)             |
    | 41    | Bottom left burner knob angular velocity              | -Inf     | Inf      | knob_Joint_2                             | hinge      | angular velocity (rad/s)   |
    | 42    | Opening linear velocity of the bottom left burner     | -Inf     | Inf      | bottom_left_burner                       | slide      | velocity (m/s)             |
    | 43    | Top right burner knob angular velocity                | -Inf     | Inf      | knob_Joint_3                             | hinge      | angular velocity (rad/s)   |
    | 44    | Opening linear velocity of the top right burner       | -Inf     | Inf      | top_right_burner                         | slide      | velocity (m/s)             |
    | 45    | Top left burner knob angular velocity                 | -Inf     | Inf      | knob_Joint_4                             | hinge      | angular velocity (rad/s)   |
    | 46    | Opening linear velocity of the top left burner        | -Inf     | Inf      | top_left_burner                          | slide      | velocity (m/s)             |
    | 47    | Angular velocity of the overhead light switch         | -Inf     | Inf      | light_switch                             | slide      | velocity (m/s)             |
    | 48    | Opening linear velocity of the overhead light         | -Inf     | Inf      | light_joint                              | hinge      | angular velocity (rad/s)   |
    | 49    | Linear velocity of the slide cabinet joint            | -Inf     | Inf      | slide_cabinet                            | slide      | velocity (m/s)             |
    | 50    | Angular velocity of the left hinge cabinet joint      | -Inf     | Inf      | left_hinge_cabinet                       | hinge      | angular velocity (rad/s)   |
    | 51    | Angular velocity of the right hinge cabinet joint     | -Inf     | Inf      | right_hinge_cabinet                      | hinge      | angular velocity (rad/s)   |
    | 52    | Anular velocity of the microwave door joint           | -Inf     | Inf      | microwave                                | hinge      | angular velocity (rad/s)   |
    | 53    | Kettle's x linear velocity                            | -Inf     | Inf      | kettle                                   | free       | linear velocity (m/s)      |
    | 54    | Kettle's y linear velocity                            | -Inf     | Inf      | kettle                                   | free       | linear velocity (m/s)      |
    | 55    | Kettle's z linear velocity                            | -Inf     | Inf      | kettle                                   | free       | linear velocity (m/s)      |
    | 56    | Kettle's x axis angular rotation                      | -Inf     | Inf      | kettle                                   | free       | angular velocity(rad/s)    |
    | 57    | Kettle's y axis angular rotation                      | -Inf     | Inf      | kettle                                   | free       | angular velocity(rad/s)    |
    | 58    | Kettle's z axis angular rotation                      | -Inf     | Inf      | kettle                                   | free       | angular velocity(rad/s)    |

    * `desired_goal`: this key represents the final goal to be achieved. The value is another `Dict` space with keys the tasks to be completed in the episode and values the joint
    goal configuration of each joint in the task as specified in the `Goal` section.

    * `achieved_goal`: this key represents the current state of the tasks. The value is another `Dict` space with keys the tasks to be completed in the episode and values the
    current joint configuration of each joint in the task.

    ## Info

    The environment also returns an `info` dictionary in each Gymnasium step. The keys are:

    - `tasks_to_complete` (list[str]): list of tasks that haven't yet been completed in the current episode.
    - `step_task_completions` (list[str]): list of tasks completed in the step taken.
    - `episode_task_completions` (list[str]): list of tasks completed during the episode uptil the current step.

    ## Rewards

    The environment's reward is `sparser`. The reward in each Gymnasium step is equal to the number of task completed in the given step. If no task is completed the returned reward will be zero.
    The tasks are considered completed when their joint configuration is within a norm threshold of `0.3` with respect to the goal configuration specified in the `Goal` section.

    ## Starting State

    The simulation starts with all of the joint position actuators of the Franka robot set to zero. The doors of the microwave and cabinets are closed, the burners turned off, and the light switch also off. The kettle
    will be placed in the bottom left burner.

    ## Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 280 timesteps.
    The episode is `terminated` when all the tasks have been completed unless the `terminate_on_tasks_completed` argument is set to `False`.

    ## Arguments

    The following arguments can be passed when initializing the environment with `gymnasium.make` kwargs:

    | Parameter                      | Type            | Default                                     | Description                                                                                                                                                               |
    | -------------------------------| --------------- | ------------------------------------------- | ----------------------------------------------------------------------------------- |
    | `tasks_to_complete`            | **list[str]**   | All possible goal tasks. Go to Goal section | The goal tasks to reach in each episode                                             |
    | `terminate_on_tasks_completed` | **bool**        | `True`                                      | Terminate episode if no more tasks to complete (episodic multitask)                 |
    | `remove_task_when_completed`   | **bool**        | `True`                                      | Remove the completed tasks from the info dictionary returned after each step        |
    | `object_noise_ratio`           | **float**       | `0.0005`                                    | Scaling factor applied to the uniform noise added to the kitchen object observations|
    | `ik_controller`                | **bool**        | `True`                                      | Use inverse kinematic (IK) controller or joint position controller                  |
    | `control_steps`                | **integer**     | `5`                                         | Number of steps to iterate the DLS in the IK controller                             |
    | `robot_noise_ratio`            | **float**       | `0.01`                                      | Scaling factor applied to the uniform noise added to the robot joint observations   |
    | `max_episode_steps`            | **integer**     | `280`                                       | Maximum number of steps per episode                                                 |

    ## Version History

    * v1: refactor version of the D4RL environment, also create dependency on newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v0: legacy versions in the [D4RL](https://github.com/Farama-Foundation/D4RL).
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 10,
    }

    def __init__(
        self,
        tasks_to_complete: "list[str]" = list(OBS_ELEMENT_GOALS.keys()),
        terminate_on_tasks_completed: bool = True,
        remove_task_when_completed: bool = True,
        object_noise_ratio: float = 0.0005,
        **kwargs,
    ):
        self.previous_obs = None
        self.robot_env = FrankaRobot(
            model_path="../assets/kitchen_franka/kitchen_assets/kitchen_env_model.xml",
            **kwargs,
        )

        self.robot_env.init_qpos[:7] = np.array(
            [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.0]
        )
        self.model = self.robot_env.model
        self.data = self.robot_env.data
        self.render_mode = self.robot_env.render_mode

        self.terminate_on_tasks_completed = terminate_on_tasks_completed
        self.remove_task_when_completed = remove_task_when_completed

        self.goal = {}
        self.tasks_to_complete = set(tasks_to_complete)
        # Validate list of tasks to complete
        for task in tasks_to_complete:
            if task not in OBS_ELEMENT_GOALS.keys():
                raise ValueError(
                    f"The task {task} cannot be found the the list of possible goals: {OBS_ELEMENT_GOALS.keys()}"
                )
            else:
                self.goal[task] = OBS_ELEMENT_GOALS[task]

        self.step_task_completions = (
            []
        )  # Tasks completed in the current environment step
        self.episode_task_completions = (
            []
        )  # Tasks completed that have been completed in the current episode
        self.object_noise_ratio = (
            object_noise_ratio  # stochastic noise added to the object observations
        )

        robot_obs = self.robot_env._get_obs()
        obs = self._get_obs(robot_obs, task)

        assert (
            int(np.round(1.0 / self.robot_env.dt)) == self.metadata["render_fps"]
        ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        self.action_space = self.robot_env.action_space
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Dict(
                    {
                        task: spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=goal.shape,
                            dtype="float64",
                        )
                        for task, goal in obs["achieved_goal"].items()
                    }
                ),
                achieved_goal=spaces.Dict(
                    {
                        task: spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=goal.shape,
                            dtype="float64",
                        )
                        for task, goal in obs["achieved_goal"].items()
                    }
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"
                ),
            )
        )

        self.init_obj_pos = dict()
        self.init_right_pad = None
        self.init_left_pad = None
        self.init_tcp = None

        EzPickle.__init__(
            self,
            tasks_to_complete,
            terminate_on_tasks_completed,
            remove_task_when_completed,
            object_noise_ratio,
            **kwargs,
        )

    def compute_reward(
        self,
        achieved_goal: "dict[str, np.ndarray]",
        desired_goal: "dict[str, np.ndarray]",
        info: "dict[str, Any]",
        action: np.ndarray,
    ):
        task = list(self.goal.keys())[0]
        reward = 0.0

        if 'microwave' in task:
            theta = self.data.joint('microwave').qpos
            reward_grab = KitchenEnv._reward_grab_effort(action)
            reward_steps = self._reward_pos(self.data.site('microhandle_site').xpos, theta, task)
            reward = sum((2.0 * hamacher_product(reward_steps[0], reward_grab),
                          8.0 * reward_steps[1]))
            if np.linalg.norm(self.data.site('microhandle_site').xpos - OBS_ELEMENT_GOALS[task]) <= 0.05:
                reward = 10
        elif 'left_hinge_cabinet' in task:
            theta = self.data.joint('left_hinge_cabinet').qpos
            reward_grab = KitchenEnv._reward_grab_effort(action)
            reward_steps = self._reward_pos(self.data.site('hinge_site1').xpos, theta, task)
            reward = sum((2.0 * hamacher_product(reward_steps[0], reward_grab),
                          8.0 * reward_steps[1]))
            if np.linalg.norm(self.data.site('hinge_site1').xpos - OBS_ELEMENT_GOALS[task]) <= 0.05:
                reward = 10
        elif 'right_hinge_cabinet' in task:
            theta = self.data.joint('right_hinge_cabinet').qpos
            reward_grab = KitchenEnv._reward_grab_effort(action)
            reward_steps = self._reward_pos(self.data.site('hinge_site2').xpos, theta, task)
            reward = sum((2.0 * hamacher_product(reward_steps[0], reward_grab),
                          8.0 * reward_steps[1]))
            if np.linalg.norm(self.data.site('hinge_site2').xpos - OBS_ELEMENT_GOALS[task]) <= 0.05:
                reward = 10
        elif 'slide_cabinet' in task:
            goal = OBS_ELEMENT_GOALS[task] # [0.332 0.607 2.6  ] if going with slide site
            obj = self.data.site('slide_site').xpos
            obj_init = np.array([-0.108, 0.607, 2.6])
            gripper_distance_apart = np.linalg.norm(
                self.data.body('right_finger').xpos - self.data.body('left_finger').xpos)
            gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
            tcp_center = (self.data.body('right_finger').xpos + self.data.body('left_finger').xpos) / 2.0
            tcp_opened = gripper_distance_apart
            tcp_to_obj = np.linalg.norm(obj - tcp_center)
            target_to_obj = np.linalg.norm(obj - goal)
            target_to_obj_init = np.linalg.norm(obj_init - goal)

            in_place = tolerance(
                target_to_obj,
                bounds=(0, BONUS_THRESH),
                margin=target_to_obj_init,
                sigmoid='long_tail',
            )

            object_grasped = self.gripper_caging_reward(
                action,
                obj,
                obj_init_pos=obj_init,
                object_reach_radius=0.01,
                obj_radius=0.015,
                pad_success_thresh=0.05,
                xz_thresh=0.005,
                high_density=True
            )
            reward = 2 * object_grasped

            if tcp_to_obj < 0.02 and tcp_opened > 0:
                reward += 1. + reward + 5. * in_place
            if target_to_obj < 0.05:
                reward = 10.

        elif 'burner' in task: # reward function from dial environment
            target = OBS_ELEMENT_GOALS[task]
            site = None
            if 'bottom_right_burner' == task:
                site = 'knob1_test'
            elif 'bottom_left_burner' == task:
                site = 'knob2_test'
            elif 'top_right_burner' == task:
                site = 'knob3_test'
            elif 'top_left_burner' == task:
                site = 'knob4_test'
            obj = self.data.site(site).xpos
            dial_push_position = obj + np.array([0.05, 0.02, 0.09])
            tcp = (self.data.body('right_finger').xpos + self.data.body('left_finger').xpos) / 2.0

            target_to_obj = np.linalg.norm(obj - target)
            target_to_obj_init = np.linalg.norm(dial_push_position - target)
            in_place = tolerance(
                target_to_obj,
                bounds=(0, BONUS_THRESH),
                margin=abs(target_to_obj_init - BONUS_THRESH),
                sigmoid='long_tail',
            )

            dial_reach_radius = 0.005
            tcp_to_obj = np.linalg.norm(dial_push_position - tcp)
            tcp_to_obj_init = np.linalg.norm(dial_push_position - self.init_tcp)
            reach = tolerance(
                tcp_to_obj,
                bounds=(0, dial_reach_radius),
                margin=abs(tcp_to_obj_init - dial_reach_radius),
                sigmoid='gaussian',
            )
            gripper_closed = min(max(0, action[-1]), 1)

            reach = hamacher_product(reach, gripper_closed)

            reward = (10 * hamacher_product(reach, in_place))
            if target_to_obj < 0.025:
                reward = 10
        elif task == 'kettle': # rewards from pick place
            _TARGET_RADIUS = 0.05
            tcp = (self.data.body('right_finger').xpos + self.data.body('left_finger').xpos) / 2.0
            obj = self.data.body('kettle').xpos
            target = OBS_ELEMENT_GOALS[task]
            obj_init_pos = np.array([-0.269, 0.35, 1.626])
            obj_to_target = np.linalg.norm(obj - target)
            tcp_to_obj = np.linalg.norm(obj - tcp)
            in_place_margin = (np.linalg.norm(obj_init_pos - target))
            gripper_distance_apart = \
                np.linalg.norm(self.data.body('right_finger').xpos - self.data.body('left_finger').xpos)
            tcp_opened = np.clip(gripper_distance_apart / 0.1, 0., 1.)
            in_place = tolerance(obj_to_target,
                                  bounds=(0, _TARGET_RADIUS),
                                  margin=in_place_margin,
                                  sigmoid='long_tail', )

            object_grasped = self.gripper_caging_reward_pick_place(action, obj)
            in_place_and_object_grasped = hamacher_product(object_grasped, in_place)
            reward = in_place_and_object_grasped

            if tcp_to_obj < 0.02 and (tcp_opened > 0) and (obj[2] - 0.01 > obj_init_pos[2]):
                reward += 1. + 5. * in_place
            if obj_to_target < _TARGET_RADIUS:
                reward = 10.
        elif 'switch' in task:
            _TARGET_RADIUS = 0.01
            obj = self.data.site('light_site').xpos
            target = OBS_ELEMENT_GOALS[task]

            obj_to_target = np.linalg.norm(obj - target)
            in_place_margin = \
                np.linalg.norm(np.array([-0.3685, 0.6157, 2.28]) - target)  # np.array is original switch pos
            in_place = tolerance(obj_to_target,
                                      bounds=(0, _TARGET_RADIUS),
                                      margin=in_place_margin,
                                      sigmoid='long_tail', )

            object_grasped = self._gripper_caging_reward_slide(action, obj, 0.02)
            in_place_and_object_grasped = hamacher_product(object_grasped, in_place)

            reward = (2 * object_grasped) + (6 * in_place_and_object_grasped)

            if obj_to_target <= _TARGET_RADIUS:
                reward = 10.
        return reward

    def _get_obs(self, robot_obs, task):
        # ok so to make the obs more like metaworld we need
        # current (hand position, gripper open/close, obj1 xyz & quaternion, obj2 xyz & quaternion)
        # previous (hand position, gripper open/close, obj1 xyz & quaternion, obj2 xyz & quaternion)
        # goal
        '''
        obj_qpos = self.data.qpos[9:].copy()
        obj_qvel = self.data.qvel[9:].copy()

        # Simulate observation noise
        obj_qpos += self.object_noise_ratio * self.robot_env.np_random.uniform(
            low=-1.0, high=1.0, size=obj_qpos.shape
        )
        obj_qvel += self.object_noise_ratio * self.robot_env.np_random.uniform(
            low=-1.0, high=1.0, size=obj_qvel.shape
        )'''

        achieved_goal = {
            task: get_joint_qpos(self.model, self.data, task).copy()
            for task in self.goal.keys()
        }

        hand_pos = self.data.body('hand').xpos
        finger_right, finger_left = (self.data.body('right_finger').xpos, self.data.body('left_finger').xpos)
        gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
        obj_pos = None
        obj_quat = None
        if task != 'kettle':
            obj_pos = self.data.site(OBS_ELEMENT_SITES[task]).xpos
            obj_quat = np.zeros(4)
            mujoco.mju_mat2Quat(obj_quat, self.data.site(OBS_ELEMENT_SITES[task]).xmat)
        else:
            obj_pos = self.data.body('kettle').xpos
            obj_quat = self.data.body('kettle').xquat

        current_obs = np.concatenate([hand_pos, [gripper_distance_apart], obj_pos, obj_quat, np.zeros(7)])

        if self.previous_obs is None:
            self.previous_obs = current_obs

        goal = OBS_ELEMENT_GOALS[task]

        obs = {
            "observation": np.concatenate([current_obs, self.previous_obs, goal]),
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal,
        }
        self.previous_obs = current_obs
        return obs

    def step(self, action, task):
        robot_obs, _, terminated, truncated, info = self.robot_env.step(action)
        obs = self._get_obs(robot_obs, task)
        info = {"tasks_to_complete": self.tasks_to_complete}

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info, action)

        if self.remove_task_when_completed:
            # When the task is accomplished remove from the list of tasks to be completed
            [
                self.tasks_to_complete.remove(element)
                for element in self.step_task_completions
            ]

        info["step_task_completions"] = self.step_task_completions
        self.episode_task_completions += self.step_task_completions
        info["episode_task_completions"] = self.episode_task_completions
        info['success'] = 1 if reward == 10 else 0
        self.step_task_completions.clear()
        if self.terminate_on_tasks_completed:
            # terminate if there are no more tasks to complete
            terminated = len(self.episode_task_completions) == len(self.goal.keys())

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.episode_task_completions.clear()
        robot_obs, _ = self.robot_env.reset(seed=seed)
        task = list(self.goal.keys())[0]
        obs = self._get_obs(robot_obs, task)
        self.task_to_complete = self.goal.copy()

        self.init_obj_pos[task] = get_joint_qpos(self.model, self.data, task)
        self.init_right_pad = self.data.body('right_finger').xpos
        self.init_left_pad = self.data.body('left_finger').xpos
        self.init_tcp = (self.data.body('right_finger').xpos + self.data.body('left_finger').xpos)/2.0
        info = {
            "tasks_to_complete": self.task_to_complete,
            "episode_task_completions": [],
            "step_task_completions": [],
        }

        return obs, info

    def render(self, **kwargs):
        return self.robot_env.render(**kwargs)

    def close(self):
        self.robot_env.close()

    def _get_pos_objects(self, task):
        name_jnt = None
        name_bdy = None
        if task == 'bottom_right_burner':
            name_jnt = 'knob_Joint_1'
            name_bdy = 'knob 1'

        dial_center = self.data.body(name_bdy).xpos
        dial_angle_rad = self.data.joint(name_jnt).qpos
        offset = np.array([
            np.sin(dial_angle_rad),
            -np.cos(dial_angle_rad),
            0
        ])
        dial_radius = 0.05

        offset *= dial_radius

        return dial_center + offset

    def _gripper_caging_reward_slide(self, action, obj_position, obj_radius):
        pad_success_margin = 0.05
        grip_success_margin = obj_radius + 0.01
        x_z_success_margin = 0.005

        tcp = (self.data.body('right_finger').xpos + self.data.body('left_finger').xpos) / 2.0
        obj_init_pos = np.array([-0.3685, 0.6157, 2.28])

        left_pad = self.data.body('left_finger').xpos
        right_pad = self.data.body('right_finger').xpos
        delta_object_y_left_pad = left_pad[1] - obj_position[1]
        delta_object_y_right_pad = obj_position[1] - right_pad[1]
        right_caging_margin = abs(abs(obj_position[1] - self.init_right_pad[1]) - pad_success_margin)
        left_caging_margin = abs(abs(obj_position[1] - self.init_left_pad[1]) - pad_success_margin)

        right_caging = tolerance(delta_object_y_right_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=right_caging_margin,
            sigmoid='long_tail',
        )
        left_caging = tolerance(delta_object_y_left_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=left_caging_margin,
            sigmoid='long_tail',
        )

        right_gripping = tolerance(delta_object_y_right_pad,
            bounds=(obj_radius, grip_success_margin),
            margin=right_caging_margin,
            sigmoid='long_tail',
        )
        left_gripping = tolerance(delta_object_y_left_pad,
            bounds=(obj_radius, grip_success_margin),
            margin=left_caging_margin,
            sigmoid='long_tail',
        )

        assert right_caging >= 0 and right_caging <= 1
        assert left_caging >= 0 and left_caging <= 1

        y_caging = hamacher_product(right_caging, left_caging)
        y_gripping = hamacher_product(right_gripping, left_gripping)

        assert y_caging >= 0 and y_caging <= 1

        tcp_xz = tcp + np.array([0., -tcp[1], 0.])
        obj_position_x_z = np.copy(obj_position) + np.array([0., -obj_position[1], 0.])
        tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)
        init_obj_x_z = obj_init_pos + np.array([0., -obj_init_pos[1], 0.])
        init_tcp_x_z = self.init_tcp + np.array([0., -self.init_tcp[1], 0.])

        tcp_obj_x_z_margin = np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin
        x_z_caging = tolerance(tcp_obj_norm_x_z,
                                bounds=(0, x_z_success_margin),
                                margin=tcp_obj_x_z_margin,
                                sigmoid='long_tail',)

        assert right_caging >= 0 and right_caging <= 1
        gripper_closed = min(max(0, action[-1]), 1)
        assert gripper_closed >= 0 and gripper_closed <= 1
        caging = hamacher_product(y_caging, x_z_caging)
        assert caging >= 0 and caging <= 1

        if caging > 0.95:
            gripping = y_gripping
        else:
            gripping = 0.
        assert gripping >= 0 and gripping <= 1

        caging_and_gripping = (caging + gripping) / 2
        assert caging_and_gripping >= 0 and caging_and_gripping <= 1

        return caging_and_gripping
    def gripper_caging_reward(self, action, obj_position, obj_radius,
                               pad_success_thresh,
                               object_reach_radius,
                               xz_thresh,
                               obj_init_pos,
                               desired_gripper_effort=1.0,
                               high_density=False,
                               medium_density=False):

        if high_density and medium_density:
            raise ValueError("Can only be either high_density or medium_density")
            # MARK: Left-right gripper information for caging reward----------------
        left_pad = self.data.body('left_finger').xpos
        right_pad = self.data.body('right_finger').xpos

        # get current positions of left and right pads (Y axis)
        pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
        # compare *current* pad positions with *current* obj position (Y axis)
        pad_to_obj_lr = np.abs(pad_y_lr - obj_position[1])
        # compare *current* pad positions with *initial* obj position (Y axis)
        pad_to_objinit_lr = np.abs(pad_y_lr - obj_init_pos[1])

        # Compute the left/right caging rewards. This is crucial for success,
        # yet counterintuitive mathematically because we invented it
        # accidentally.
        #
        # Before touching the object, `pad_to_obj_lr` ("x") is always separated
        # from `caging_lr_margin` ("the margin") by some small number,
        # `pad_success_thresh`.
        #
        # When far away from the object:
        #       x = margin + pad_success_thresh
        #       --> Thus x is outside the margin, yielding very small reward.
        #           Here, any variation in the reward is due to the fact that
        #           the margin itself is shifting.
        # When near the object (within pad_success_thresh):
        #       x = pad_success_thresh - margin
        #       --> Thus x is well within the margin. As long as x > obj_radius,
        #           it will also be within the bounds, yielding maximum reward.
        #           Here, any variation in the reward is due to the gripper
        #           moving *too close* to the object (i.e, blowing past the
        #           obj_radius bound).
        #
        # Therefore, before touching the object, this is very nearly a binary
        # reward -- if the gripper is between obj_radius and pad_success_thresh,
        # it gets maximum reward. Otherwise, the reward very quickly falls off.
        #
        # After grasping the object and moving it away from initial position,
        # x remains (mostly) constant while the margin grows considerably. This
        # penalizes the agent if it moves *back* toward `obj_init_pos`, but
        # offers no encouragement for leaving that position in the first place.
        # That part is left to the reward functions of individual environments.
        caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
        caging_lr = [tolerance(
            pad_to_obj_lr[i],  # "x" in the description above
            bounds=(obj_radius, pad_success_thresh),
            margin=caging_lr_margin[i],  # "margin" in the description above
            sigmoid='long_tail',
        ) for i in range(2)]
        caging_y = hamacher_product(*caging_lr)

        # MARK: X-Z gripper information for caging reward-----------------------
        tcp = (self.data.body('right_finger').xpos + self.data.body('left_finger').xpos)/2.0
        xz = [0, 2]

        # Compared to the caging_y reward, caging_xz is simple. The margin is
        # constant (something in the 0.3 to 0.5 range) and x shrinks as the
        # gripper moves towards the object. After picking up the object, the
        # reward is maximized and changes very little
        caging_xz_margin = np.linalg.norm(obj_init_pos[xz] - self.init_tcp[xz])
        caging_xz_margin -= xz_thresh
        caging_xz = tolerance(
            np.linalg.norm(tcp[xz] - obj_position[xz]),  # "x" in the description above
            bounds=(0, xz_thresh),
            margin=caging_xz_margin,  # "margin" in the description above
            sigmoid='long_tail',
        )

        # MARK: Closed-extent gripper information for caging reward-------------
        gripper_closed = min(max(0, action[-1]), desired_gripper_effort) \
                         / desired_gripper_effort

        # MARK: Combine components----------------------------------------------
        caging = hamacher_product(caging_y, caging_xz)
        gripping = gripper_closed if caging > 0.97 else 0.
        caging_and_gripping = hamacher_product(caging, gripping)

        if high_density:
            caging_and_gripping = (caging_and_gripping + caging) / 2
        if medium_density:
            tcp = (self.data.body('right_finger').xpos + self.data.body('left_finger').xpos)/2.0
            tcp_to_obj = np.linalg.norm(obj_position - tcp)
            tcp_to_obj_init = np.linalg.norm(obj_init_pos - self.init_tcp)
            # Compute reach reward
            # - We subtract `object_reach_radius` from the margin so that the
            #   reward always starts with a value of 0.1
            reach_margin = abs(tcp_to_obj_init - object_reach_radius)
            reach = tolerance(
                tcp_to_obj,
                bounds=(0, object_reach_radius),
                margin=reach_margin,
                sigmoid='long_tail',
            )
            caging_and_gripping = (caging_and_gripping + reach) / 2

        return caging_and_gripping

    def gripper_caging_reward_pick_place(self, action, obj_position):
        pad_success_margin = 0.05
        x_z_success_margin = 0.005
        obj_radius = 0.015
        tcp = (self.data.body('right_finger').xpos + self.data.body('left_finger').xpos) / 2.0
        obj_init_pos = np.array([-0.269, 0.35, 1.626])

        left_pad = self.data.body('left_finger').xpos
        right_pad = self.data.body('right_finger').xpos
        delta_object_y_left_pad = left_pad[1] - obj_position[1]
        delta_object_y_right_pad = obj_position[1] - right_pad[1]
        right_caging_margin = abs(abs(obj_position[1] - self.init_right_pad[1])
            - pad_success_margin)
        left_caging_margin = abs(abs(obj_position[1] - self.init_left_pad[1])
            - pad_success_margin)

        right_caging = tolerance(delta_object_y_right_pad,
                                bounds=(obj_radius, pad_success_margin),
                                margin=right_caging_margin,
                                sigmoid='long_tail',)
        left_caging = tolerance(delta_object_y_left_pad,
                                bounds=(obj_radius, pad_success_margin),
                                margin=left_caging_margin,
                                sigmoid='long_tail',)

        y_caging = hamacher_product(left_caging,
                                                 right_caging)

        # compute the tcp_obj distance in the x_z plane
        tcp_xz = tcp + np.array([0., -tcp[1], 0.])
        obj_position_x_z = np.copy(obj_position) + np.array([0., -obj_position[1], 0.])
        tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)

        # used for computing the tcp to object object margin in the x_z plane
        init_obj_x_z = obj_init_pos + np.array([0., -obj_init_pos[1], 0.])
        init_tcp_x_z = self.init_tcp + np.array([0., -self.init_tcp[1], 0.])
        tcp_obj_x_z_margin = np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin

        x_z_caging = tolerance(tcp_obj_norm_x_z,
                                bounds=(0, x_z_success_margin),
                                margin=tcp_obj_x_z_margin,
                                sigmoid='long_tail',)

        gripper_closed = min(max(0, action[-1]), 1)
        caging = hamacher_product(y_caging, x_z_caging)

        gripping = gripper_closed if caging > 0.97 else 0.
        caging_and_gripping = hamacher_product(caging, gripping)
        caging_and_gripping = (caging_and_gripping + caging) / 2
        return caging_and_gripping

    @staticmethod
    def _reward_grab_effort(actions):
        return (np.clip(actions[-1], -1, 1) + 1.0) / 2.0

    def _reward_pos(self, obs, theta, task):
        hand = self.data.body('hand').xpos
        door = None
        if 'microwave' in task:
            door = self.data.site('microhandle_site').xpos  # + np.array([-0.05, 0, 0])
        elif 'left_hinge_cabinet' in task:
            door = self.data.site('hinge_site1').xpos  # + np.array([-0.05, 0, 0])
        elif 'right_hinge_cabinet' in task:
            door = self.data.site('hinge_site2').xpos  # + np.array([-0.05, 0, 0])
        threshold = 0.12
        # floor is a 3D funnel centered on the door handle
        '''radius = np.linalg.norm(hand[:2] - door[:2])
        if radius <= threshold:
            floor = 0.0
        else:
            floor = 0.04 * np.log(radius - threshold) + 0.4
        # prevent the hand from running into the handle prematurely by keeping
        # it above the "floor"
        '''
        above_floor = 1.0

        ''''if hand[2] >= floor else tolerance(
            floor - hand[2],
            bounds=(0.0, 0.01),
            margin=floor / 2.0,
            sigmoid='long_tail',
        )'''
        ''''# move the hand to a position between the handle and the main door body
        in_place = tolerance(
            np.linalg.norm(hand - door - np.array([0.05, 0.03, -0.01])),
            bounds=(0, threshold / 2.0),
            margin=0.5,
            sigmoid='long_tail',
        )'''
        # grab the door's handle
        in_place = tolerance(
            np.linalg.norm(hand - door),
            bounds=(0, 0.02),
            margin=0.5,
            sigmoid='long_tail',
        )
        ready_to_open = hamacher_product(above_floor, in_place)

        # now actually open the door
        door_angle = -theta
        a = 0.4  # Relative importance of just *trying* to open the door at all
        b = 0.6  # Relative importance of fully opening the door
        opened = a * float(theta < -np.pi/90.) + b * tolerance(
            np.pi/2. + np.pi/6 - door_angle,
            bounds=(0, 0.5),
            margin=np.pi/3.,
            sigmoid='long_tail',
        )[0]
        return ready_to_open, opened

def tolerance(x,
              bounds=(0.0, 0.0),
              margin=0.0,
              sigmoid='gaussian',
              value_at_margin=0.1):
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
        x: A scalar or numpy array.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Float. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
        'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
        value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
        A float or numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError('Lower bound must be <= upper bound.')
    if margin < 0:
        raise ValueError('`margin` must be non-negative. Current value: {}'.format(margin))

    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin,
                                                   sigmoid))

    return float(value) if np.isscalar(x) else value

def _sigmoids(x, value_at_1, sigmoid):
    """Returns 1 when `x` == 0, between 0 and 1 otherwise.

    Args:
        x: A scalar or numpy array.
        value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
        sigmoid: String, choice of sigmoid type.

    Returns:
        A numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
        `quadratic` sigmoids which allow `value_at_1` == 0.
        ValueError: If `sigmoid` is of an unknown type.
    """
    if sigmoid in ('cosine', 'linear', 'quadratic'):
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                '`value_at_1` must be nonnegative and smaller than 1, '
                'got {}.'.format(value_at_1))
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                             'got {}.'.format(value_at_1))

    if sigmoid == 'gaussian':
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x * scale)**2)

    elif sigmoid == 'hyperbolic':
        scale = np.arccosh(1 / value_at_1)
        return 1 / np.cosh(x * scale)

    elif sigmoid == 'long_tail':
        scale = np.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale)**2 + 1)

    elif sigmoid == 'reciprocal':
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == 'cosine':
        scale = np.arccos(2 * value_at_1 - 1) / np.pi
        scaled_x = x * scale
        return np.where(
            abs(scaled_x) < 1, (1 + np.cos(np.pi * scaled_x)) / 2, 0.0)

    elif sigmoid == 'linear':
        scale = 1 - value_at_1
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == 'quadratic':
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

    elif sigmoid == 'tanh_squared':
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - np.tanh(x * scale)**2

    else:
        raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))

def hamacher_product(a, b):
    """The hamacher (t-norm) product of a and b.

    computes (a * b) / ((a + b) - (a * b))

    Args:
        a (float): 1st term of hamacher product.
        b (float): 2nd term of hamacher product.
    Raises:
        ValueError: a and b must range between 0 and 1

    Returns:
        float: The hammacher product of a and b
    """
    if not ((0. <= a <= 1.) and (0. <= b <= 1.)):
        raise ValueError("a and b must range between 0 and 1")

    denominator = a + b - (a * b)
    h_prod = ((a * b) / denominator) if denominator > 0 else 0

    assert 0. <= h_prod <= 1.
    return h_prod
