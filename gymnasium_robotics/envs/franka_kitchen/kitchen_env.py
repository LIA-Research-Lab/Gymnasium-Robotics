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

OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
'''OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}'''
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.085, 0.6409, 2.22646769]),
    "bottom_left_burner": np.array([-0.208, 0.6409, 2.22646769]),
    "top burner": np.array([-0.085, 0.6409, 2.34046769]),
    "top_left_burner": np.array([-0.208, 0.6409, 2.34046769]),
    "light switch": np.array([-0.42413573, 0.61301656, 2.28]),
    "slide cabinet": np.array([0.292, 0.607, 2.6]),
    "left_hinge_cabinet": np.array([-1.0265508, 0.38476558, 2.6]),
    "hinge cabinet": np.array([-0.16461378,  0.3874147, 2.6]),
    "microwave": np.array([-1.05611982, -0.0269469, 1.792]),
    "kettle": np.array([-0.23,  0.75,  1.62]),
}
OBS_ELEMENT_SITES = {
    "bottom burner": 'knob1_site',
    "bottom_left_burner": 'knob2_test',
    "top burner": 'knob3_site',
    "top_left_burner": 'knob4_test',
    "light switch": 'light_site',
    "slide cabinet": 'slide_site',
    "left_hinge_cabinet": 'hinge_site1',
    "hinge cabinet": 'hinge_site2',
    "microwave": 'microhandle_site',
    "kettle": 'kettle_site',
}

BONUS_THRESH = 0.3


class KitchenEnv(GoalEnv, EzPickle):
    """
    ## Description

    This environment was introduced in ["Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning"](https://arxiv.org/abs/1910.11956)
    by Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, Karol Hausman.

    The environment is based on the 9 degrees of freedom [Franka robot](https://www.franka.de/). The Franka robot is placed in a kitchen environment containing several common
    household items: a microwave, a kettle, an overhead light, cabinets, and an oven. The environment is a `multitask` goal in which the robot has to interact with the previously
    mentioned items in order to reach a desired goal configuration. For example, one such state is to have the microwave and sliding cabinet door open with the kettle on the top burner
    and the overhead light on. The goal tasks can be configured when the environment is created.

    ## Goal

    The goal has a multitask configuration. The multiple tasks to be completed in an episode can be set by passing a list of tasks to the argument`tasks_to_complete`. For example, to open
    the microwave door and move the kettle create the environment as follows:

    ```python
    import gymnasium as gym
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'])
    ```

    The following is a table with all the possible tasks and their respective joint goal values:

    | Task             | Description                                                    | Joint Type | Goal                                     |
    | ---------------- | -------------------------------------------------------------- | ---------- | ---------------------------------------- |
    | "bottom burner"  | Turn the oven knob that activates the bottom burner            | slide      | [-0.88, -0.01]                           |
    | "top burner"     | Turn the oven knob that activates the top burner               | slide      | [-0.92, -0.01]                           |
    | "light switch"   | Turn on the light switch                                       | slide      | [-0.69, -0.05]                           |
    | "slide cabinet"  | Open the slide cabinet                                         | slide      | 0.37                                     |
    | "hinge cabinet"  | Open the left hinge cabinet                                    | hinge      | [0.0, 1.45]                              |
    | "microwave"      | Open the microwave door                                        | hinge      | 0.37                                     |
    | "kettle"         | Move the kettle to the top left burner                         | free       | [-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06] |


    ## Action Space

    The default joint actuators in the Franka MuJoCo model are position controlled. However, the action space of the environment are joint velocities clipped between -1 and 1 rad/s.
    The space is a `Box(-1.0, 1.0, (9,), float32)`. The desired joint position control input is estimated in each time step with the current joint position values and the desired velocity
    action:

    | Num | Action                                         | Action Min  | Action Max  | Joint | Unit  |
    | --- | ---------------------------------------------- | ----------- | ----------- | ----- | ----- |
    | 0   | `robot:panda0_joint1` angular velocity         | -1          | 1           | hinge | rad/s |
    | 1   | `robot:panda0_joint2` angular velocity         | -1          | 1           | hinge | rad/s |
    | 2   | `robot:panda0_joint3` angular velocity         | -1          | 1           | hinge | rad/s |
    | 3   | `robot:panda0_joint4` angular velocity         | -1          | 1           | hinge | rad/s |
    | 4   | `robot:panda0_joint5` angular velocity         | -1          | 1           | hinge | rad/s |
    | 5   | `robot:panda0_joint6` angular velocity         | -1          | 1           | hinge | rad/s |
    | 6   | `robot:panda0_joint7` angular velocity         | -1          | 1           | hinge | rad/s |
    | 7   | `robot:r_gripper_finger_joint` linear velocity | -1          | 1           | slide | m/s   |
    | 8   | `robot:l_gripper_finger_joint` linear velocity | -1          | 1           | slide | m/s   |

    ## Observation Space

    The observation is a `goal-aware` observation space. The observation space contains the following keys:

    * `observation`: this is a `Box(-inf, inf, shape=(59,), dtype="float64")` space and it is formed by the robot's joint positions and velocities, as well as
        the pose and velocities of the kitchen items. An additional uniform noise of range `[-1,1]` is added to the observations. The noise is also scaled by a factor
        of `robot_noise_ratio` and `object_noise_ratio` given in the environment arguments. The elements of the `observation` array are the following:


    | Num   | Observation                                           | Min      | Max      | Joint Name (in corresponding XML file)   | Joint Type | Unit                       |
    | ----- | ----------------------------------------------------- | -------- | -------- | ---------------------------------------- | ---------- | -------------------------- |
    | 0     | `robot:panda0_joint1` hinge joint angle value                | -Inf     | Inf      | robot:panda0_joint1                             | hinge      | angle (rad)                |
    | 1     | `robot:panda0_joint2` hinge joint angle value                | -Inf     | Inf      | robot:panda0_joint2                             | hinge      | angle (rad)                |
    | 2     | `robot:panda0_joint3` hinge joint angle value                | -Inf     | Inf      | robot:panda0_joint3                             | hinge      | angle (rad)                |
    | 3     | `robot:panda0_joint4` hinge joint angle value                | -Inf     | Inf      | robot:panda0_joint4                             | hinge      | angle (rad)                |
    | 4     | `robot:panda0_joint5` hinge joint angle value                | -Inf     | Inf      | robot:panda0_joint5                             | hinge      | angle (rad)                |
    | 5     | `robot:panda0_joint6` hinge joint angle value                | -Inf     | Inf      | robot:panda0_joint6                             | hinge      | angle (rad)                |
    | 6     | `robot:panda0_joint7` hinge joint angle value                | -Inf     | Inf      | robot:panda0_joint7                             | hinge      | angle (rad)                |
    | 7     | `robot:r_gripper_finger_joint` slide joint translation value   | -Inf     | Inf      | robot:r_gripper_finger_joint                      | slide      | position (m)               |
    | 8     | `robot:l_gripper_finger_joint` slide joint translation value   | -Inf     | Inf      | robot:l_gripper_finger_joint                      | slide      | position (m)               |
    | 9     | `robot:panda0_joint1` hinge joint angular velocity           | -Inf     | Inf      | robot:panda0_joint1                             | hinge      | angular velocity (rad/s)   |
    | 10    | `robot:panda0_joint2` hinge joint angular velocity           | -Inf     | Inf      | robot:panda0_joint2                             | hinge      | angular velocity (rad/s)   |
    | 11    | `robot:panda0_joint3` hinge joint angular velocity           | -Inf     | Inf      | robot:panda0_joint3                             | hinge      | angular velocity (rad/s)   |
    | 12    | `robot:panda0_joint4` hinge joint angular velocity           | -Inf     | Inf      | robot:panda0_joint4                             | hinge      | angular velocity (rad/s)   |
    | 13    | `robot:panda0_joint5` hinge joint angular velocity           | -Inf     | Inf      | robot:panda0_joint5                             | hinge      | angular velocity (rad/s)   |
    | 14    | `robot:panda0_joint6` hinge joint angular velocity           | -Inf     | Inf      | robot:panda0_joint6                             | hinge      | angular velocity (rad/s)   |
    | 15    | `robot:panda0_joint7` hinge joint angular velocity           | -Inf     | Inf      | robot:panda0_joint7                             | hinge      | angle (rad)                |
    | 16    | `robot:r_gripper_finger_joint` slide joint linear velocity     | -Inf     | Inf      | robot:r_gripper_finger_joint                      | slide      | linear velocity (m/s)      |
    | 17    | `robot:l_gripper_finger_joint` slide joint linear velocity     | -Inf     | Inf      | robot:l_gripper_finger_joint                      | slide      | linear velocity (m/s)      |
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

    The environment's reward is `sparse`. The reward in each Gymnasium step is equal to the number of task completed in the given step. If no task is completed the returned reward will be zero.
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
    | `robot_noise_ratio`            | **float**       | `0.01`                                      | Scaling factor applied to the uniform noise added to the robot joint observations   |
    | `max_episode_steps`            | **integer**     | `280`                                       | Maximum number of steps per episode                                                 |

    ## Version History

    * v1: updated version with most recent python MuJoCo bindings.
    * v0: legacy versions in the [D4RL](https://github.com/Farama-Foundation/D4RL).
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 12,
    }

    def __init__(
        self,
        tasks_to_complete: "list[str]" = list(OBS_ELEMENT_GOALS.keys()),
        terminate_on_tasks_completed: bool = False,
        remove_task_when_completed: bool = True,
        object_noise_ratio: float = 0.0005,
        **kwargs,
    ):
        self.robot_env = FrankaRobot(
            model_path="../assets/kitchen_franka/kitchen_assets/kitchen_env_model.xml",
            **kwargs,
        )

        self.robot_env.init_qpos = np.array(
            [
                1.48388023e-01,
                -1.76848573e00,
                1.84390296e00,
                -2.47685760e00,
                2.60252026e-01,
                7.12533105e-01,
                1.59515394e00,
                4.79267505e-02,
                3.71350919e-02,
                -2.66279850e-04,
                -5.18043486e-05,
                3.12877220e-05,
                -4.51199853e-05,
                -3.90842156e-06,
                -4.22629655e-05,
                6.28065475e-05,
                4.04984708e-05,
                4.62730939e-04,
                -2.26906415e-04,
                -4.65501369e-04,
                -6.44129196e-03,
                -1.77048263e-03,
                1.08009684e-03,
                -2.69397440e-01,
                3.50383255e-01,
                1.61944683e00,
                1.00618764e00,
                4.06395120e-03,
                -6.62095997e-03,
                -2.68278933e-04,
            ]
        )

        self.model = self.robot_env.model
        self.data = self.robot_env.data
        self.render_mode = self.robot_env.render_mode

        self.terminate_on_tasks_completed = terminate_on_tasks_completed
        self.remove_task_when_completed = remove_task_when_completed

        self.goal = {}
        self.tasks_to_complete = list(set(tasks_to_complete))
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
        obs = self._get_obs(robot_obs)

        assert (
            int(np.round(1.0 / self.robot_env.dt)) == self.metadata["render_fps"]
        ), f'Expected value: {int(np.round(1.0 / self.robot_env.dt))}, Actual value: {self.metadata["render_fps"]}'

        self.action_space = self.robot_env.action_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(62,), dtype=np.float32
        )

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
    ):
        
        if self.tasks_to_complete[0] == 'bottom burner':
            robot_ee_pos = (self.data.body('panda0_rightfinger').xpos + self.data.body('panda0_leftfinger').xpos) / 2.0
            knob_position = self.data.site(OBS_ELEMENT_SITES[self.tasks_to_complete[0]]).xpos  # Assuming obj1 is the knob

            # 1. Distance Reward: Encourage the gripper to move close to the knob
            distance = np.linalg.norm(robot_ee_pos - knob_position)
            distance_reward = -distance  # Negative because we want to minimize distance

            # 2. Goal Position Reward: Encourage exact positioning (for simplicity, not considering orientation)
            # Assuming 'goal_position' to be the ideal position to start turning the knob
            position_error = np.linalg.norm(robot_ee_pos - OBS_ELEMENT_GOALS[self.tasks_to_complete[0]])
            position_reward = -position_error  # Negative because we want to minimize error

            # Combine the rewards
            # Weights can be adjusted based on the relative importance of each component
            total_reward = (distance_reward * 0.5) + (position_reward * 0.5)

            return total_reward
        elif self.tasks_to_complete[0] == 'top burner':
            gripper_pos = (self.data.body('panda0_rightfinger').xpos + self.data.body('panda0_leftfinger').xpos) / 2.0
            knob_pos = self.data.site(OBS_ELEMENT_SITES[self.tasks_to_complete[0]]).xpos  # Assuming obj1 is the oven knob
    
            # 1. Reward for being close to the knob
            distance_to_knob = np.linalg.norm(gripper_pos - knob_pos)
            # Inverse of the distance to make it a positive reward, higher when closer
            proximity_reward = 1 / (1 + distance_to_knob)
    
            # 2. Reward for achieving the goal state
            # Assuming the goal is a position where the knob should be when the burner is on
            # This might involve a specific position or orientation, but only position is known
            goal_pos = OBS_ELEMENT_GOALS[self.tasks_to_complete[0]]
            distance_to_goal = np.linalg.norm(knob_pos - goal_pos)
            # Similar to proximity, reward for being close to the goal position
            goal_achievement_reward = 1 / (1 + distance_to_goal)
    
            # Total reward is a weighted sum of the proximity and goal achievement rewards
            # Weights can be adjusted based on importance of subtasks
            total_reward = 0.7 * proximity_reward + 0.3 * goal_achievement_reward

            return total_reward
        elif self.tasks_to_complete[0] == "hinge cabinet":
            DISTANCE_SCALE = -0.1
            GRASP_REWARD = 10
            OPENING_REWARD_SCALE = -0.1
    
            # Calculate the distance to the handle
            handle_position = self.data.site(OBS_ELEMENT_SITES[self.tasks_to_complete[0]]).xpos  # Assuming obj1 is the handle
            gripper_position = (self.data.body('panda0_rightfinger').xpos + self.data.body('panda0_leftfinger').xpos) / 2.0
            distance_to_handle = np.linalg.norm(gripper_position - handle_position)
            gripper_openness = np.linalg.norm(
                self.data.body('panda0_rightfinger').xpos - self.data.body('panda0_leftfinger').xpos)
            gripper_openness = np.clip(gripper_openness / 0.1, 0., 1.)
            # Reward for reducing distance to handle
            distance_reward = DISTANCE_SCALE * distance_to_handle
    
            # Check if the gripper is close enough and 'closed' to consider it grasping
            GRASP_THRESHOLD = 0.1  # Threshold for distance to consider the gripper is grasping the handle
            GRIPPER_CLOSED_VALUE = -0.5  # Assuming gripper_openness closer to -1 is more closed
            is_grasping = distance_to_handle < GRASP_THRESHOLD and gripper_openness < GRIPPER_CLOSED_VALUE
            grasp_reward = GRASP_REWARD if is_grasping else 0
    
            # Reward for moving the handle towards the goal position
            goal_position = OBS_ELEMENT_GOALS[self.tasks_to_complete[0]]
            current_handle_position = self.data.site(OBS_ELEMENT_SITES[self.tasks_to_complete[0]]).xpos
            distance_to_goal = np.linalg.norm(current_handle_position - goal_position)
            opening_reward = OPENING_REWARD_SCALE * distance_to_goal
    
            # Total reward
            total_reward = distance_reward + grasp_reward + opening_reward
    
            return total_reward
        elif self.tasks_to_complete[0] == "slide cabinet":
            gripper_pos = (self.data.body('panda0_rightfinger').xpos + self.data.body('panda0_leftfinger').xpos) / 2.0
            handle_pos = self.data.site(OBS_ELEMENT_SITES[self.tasks_to_complete[0]]).xpos
            goal_pos = OBS_ELEMENT_GOALS[self.tasks_to_complete[0]]
    
            # Compute the distance between the gripper and the handle
            distance_to_handle = np.linalg.norm(gripper_pos - handle_pos)
    
            # Compute how much the handle has moved towards the goal
            initial_distance_to_goal = np.linalg.norm(handle_pos - goal_pos)
            current_distance_to_goal = np.linalg.norm(self.data.site(OBS_ELEMENT_SITES[self.tasks_to_complete[0]]).xpos - goal_pos)
    
            # Reward for approaching the handle: more positive as the distance decreases
            # Normalized by adding a small constant to prevent division by zero.
            approach_reward = 1.0 / (distance_to_handle + 0.01)
    
            # Reward for moving handle towards the goal: more positive as the handle gets closer to the goal
            progress_reward = (initial_distance_to_goal - current_distance_to_goal) / (initial_distance_to_goal + 0.01)
    
            # Total reward combines both elements, with possible weighting
            total_reward = approach_reward + progress_reward
    
            return total_reward
        elif self.tasks_to_complete[0] == 'microwave':
            gripper_position = (self.data.body('panda0_rightfinger').xpos + self.data.body('panda0_leftfinger').xpos) / 2.0
            gripper_openness = np.linalg.norm(
                self.data.body('panda0_rightfinger').xpos - self.data.body('panda0_leftfinger').xpos)
            gripper_openness = np.clip(gripper_openness / 0.1, 0., 1.)
            handle_position = self.data.site(OBS_ELEMENT_SITES[self.tasks_to_complete[0]]).xpos  # Assuming obj1 is the microwave handle
    
            # Constants for reward scaling
            distance_scale = -1.0  # Negative because we want to minimize distance
            openness_scale = 1.0  # Positive to encourage opening when close to handle
            goal_scale = 2.0  # High reward for achieving goal state
    
            # Calculate distance to the handle
            distance_to_handle = np.linalg.norm(gripper_position - handle_position)
    
            # Calculate similarity of gripper openness to ideal grip openness
            # Assume ideal openness to grasp handle is around -0.5 (partially open)
            ideal_openness = -0.5
            openness_diff = abs(gripper_openness - ideal_openness)
    
            # Reward for approaching the handle
            reward_distance = distance_scale * distance_to_handle
    
            # Reward for having the gripper in the right openness state
            # Only significant when close to the handle
            if distance_to_handle < 0.05:  # Threshold to consider 'close' to the handle
                reward_openness = openness_scale * (1 - openness_diff)
            else:
                reward_openness = 0
    
            # Reward for achieving goal (opening the microwave)
            # This could be defined by the position of the microwave door if movable
            # Since goal_position is not clear whether it's handle's goal or door's, assume it's door's final position
             # Assuming obj2 is the microwave door
            if np.linalg.norm(handle_position - OBS_ELEMENT_GOALS[self.tasks_to_complete[0]]) < 0.01:  # Close to goal position
                reward_goal_achieved = goal_scale
            else:
                reward_goal_achieved = 0
    
            # Total reward
            total_reward = reward_distance + reward_openness + reward_goal_achieved
    
            return total_reward
        elif self.tasks_to_complete[0] == 'light switch':
            gripper_pos = (self.data.body('panda0_rightfinger').xpos + self.data.body('panda0_leftfinger').xpos) / 2.0
            switch_pos = self.data.site(OBS_ELEMENT_SITES[self.tasks_to_complete[0]]).xpos
            goal_pos = OBS_ELEMENT_GOALS[self.tasks_to_complete[0]]
    
            # Constants for reward calculation
            close_enough_threshold = 0.05  # threshold to consider gripper "close enough" to the switch
    
            # Calculate distances
            distance_to_switch = np.linalg.norm(gripper_pos - switch_pos)
            distance_to_goal = np.linalg.norm(switch_pos - goal_pos)
     
            # Define the reward components
            # Reward for getting the gripper close to the switch
            if distance_to_switch < close_enough_threshold:
                proximity_reward = 1.0  # Max reward when very close to the switch
            else:
                # Negative exponential decay to reward getting closer
                proximity_reward = np.exp(-distance_to_switch)
    
            # Reward for moving the switch closer to the goal position
            if distance_to_goal < close_enough_threshold:
                goal_achievement_reward = 1.0  # Max reward when switch is at the goal position
            else:
                # Negative exponential decay to reward progress towards the goal
                goal_achievement_reward = np.exp(-distance_to_goal)
    
            # Combine rewards
            # Weight could be adjusted based on the phase of the task; initially focus more on getting to the switch
            total_reward = 0.7 * proximity_reward + 0.3 * goal_achievement_reward
            return total_reward

        elif self.tasks_to_complete[0] == 'kettle': 
            robot_ee_pos = (self.data.body('panda0_rightfinger').xpos + self.data.body('panda0_leftfinger').xpos) / 2.0
            kettle_pos = self.data.site(OBS_ELEMENT_SITES[self.tasks_to_complete[0]]).xpos
            goal_pos = OBS_ELEMENT_GOALS[self.tasks_to_complete[0]]

            # Constants for reward computation
            DISTANCE_SCALE = 1.0
            GOAL_REACHED_REWARD = 10.0
            GRIPPER_CLOSE_REWARD = 5.0
    
            # Compute the Euclidean distance between the gripper and the kettle handle
            distance_to_kettle = np.linalg.norm(robot_ee_pos - kettle_pos)
    
            # Compute the Euclidean distance between the kettle's current position and the goal position
            distance_to_goal = np.linalg.norm(kettle_pos - goal_pos)
    
            # Reward for moving closer to the kettle
            reward = -DISTANCE_SCALE * distance_to_kettle
    
            gripper_openness = np.linalg.norm(
                self.data.body('panda0_rightfinger').xpos - self.data.body('panda0_leftfinger').xpos)
            gripper_openness = np.clip(gripper_openness / 0.1, 0., 1.)

            # If the gripper is close enough to the kettle, provide additional reward for grasping
            if distance_to_kettle < 0.05:  # threshold for being "close"
                reward += GRIPPER_CLOSE_REWARD * (1 - gripper_openness)  # encourage the gripper to close
    
            # Check if kettle is at the goal position within a certain threshold
            if distance_to_goal < 0.1:  # goal threshold
                reward += GOAL_REACHED_REWARD
    
            return reward



    def _get_obs(self, robot_obs):
        obj_qpos = self.data.qpos[9:].copy()
        obj_qvel = self.data.qvel[9:].copy()

        # Simulate observation noise
        obj_qpos += (
            self.object_noise_ratio
            * self.robot_env.robot_pos_noise_amp[8:]
            * self.robot_env.np_random.uniform(low=-1.0, high=1.0, size=obj_qpos.shape)
        )
        obj_qvel += (
            self.object_noise_ratio
            * self.robot_env.robot_vel_noise_amp[9:]
            * self.robot_env.np_random.uniform(low=-1.0, high=1.0, size=obj_qvel.shape)
        )

        achieved_goal = {
            task: self.data.qpos[OBS_ELEMENT_INDICES[task]] for task in self.goal.keys()
        }

        obs = {
            "observation": np.concatenate((robot_obs, obj_qpos, obj_qvel)),
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal,
        }

        return obs

    def step(self, action):
        robot_obs, _, terminated, truncated, info = self.robot_env.step(action)
        obs = self._get_obs(robot_obs)

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        if self.remove_task_when_completed:
            # When the task is accomplished remove from the list of tasks to be completed
            [
                self.tasks_to_complete.remove(element)
                for element in self.step_task_completions
            ]

        info = {"tasks_to_complete": list(self.tasks_to_complete)}
        info["step_task_completions"] = self.step_task_completions.copy()

        for task in self.step_task_completions:
            if task not in self.episode_task_completions:
                self.episode_task_completions.append(task)
        info["episode_task_completions"] = self.episode_task_completions
        if self.terminate_on_tasks_completed:
            # terminate if there are no more tasks to complete
            terminated = len(self.episode_task_completions) == len(self.goal.keys())
        obs = np.concatenate([obs['observation'], OBS_ELEMENT_GOALS[self.tasks_to_complete[0]]])
        info['success'] = np.linalg.norm(OBS_ELEMENT_GOALS[self.tasks_to_complete[0]] - self.data.site(OBS_ELEMENT_SITES[self.tasks_to_complete[0]]).xpos) < 0.05
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.episode_task_completions.clear()
        robot_obs, _ = self.robot_env.reset(seed=seed)
        obs = self._get_obs(robot_obs)
        self.tasks_to_complete = list(set(self.goal.keys()))
        info = {
            "tasks_to_complete": list(self.tasks_to_complete),
            "episode_task_completions": [],
            "step_task_completions": [],
        }
        obs = np.concatenate([obs['observation'], OBS_ELEMENT_GOALS[self.tasks_to_complete[0]]])
        info['success'] = np.linalg.norm(OBS_ELEMENT_GOALS[self.tasks_to_complete[0]] - self.data.site(OBS_ELEMENT_SITES[self.tasks_to_complete[0]]).xpos) < 0.05
        return obs, info

    def render(self):
        return self.robot_env.render()

    def close(self):
        self.robot_env.close()
