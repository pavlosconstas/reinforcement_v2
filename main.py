import os
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint
from scipy.stats import truncnorm
from scipy.signal import savgol_filter

class SpeechFluencyEnv(gym.Env):
    """
    Description:
        See dosing_rl/resources/Diabetic Background.ipynb
    Source:
        See dosing_rl/resources/Diabetic Background.ipynb
    Observation:
        Type: Box(9)
                                                                Min         Max
        0	Blood Glucose                                       0           Inf
        1	Remote Insulin                                      0           Inf
        2	Plasma Insulin                                      0           Inf
        3	S1                                                  0           Inf
        4	S2                                                  0           Inf
        5	Gut blood glucose                                   0           Inf
        6	Meal disturbance                                    0           Inf
        7	Previous Blood glucose                              0           Inf
        8   Previous meal disturbance                           0           Inf

    Actions:
        Type: Continuous
        Administered Insulin pump [mU/min]

    Reward:
        smooth function centered at 80 (self.target)
    Starting State:
        http://apmonitor.com/pdc/index.php/Main/DiabeticBloodGlucose
    Episode Termination:
        self.episode_length reached
    """

    def __init__(self):
        self.__version__ = "0.0.1"

        # General variables defining the environment
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.seed()

        # Lists that will hold episode data
        self.speech_fluency = None
        self.depression = None
        self.anxiety = None
        self.insomnia = None
        self.medication_effect = None
        self.day = None
        self.medication_dose = None

        # Defining possible actions (medication dose)
        self.action_space = spaces.Box(0.0, 10.0, shape=(1,))

        # Defining observation space
        lows = np.zeros(5)
        highs = np.ones(5) * np.inf
        self.observation_space = spaces.Box(lows, highs)

        # Reward definitions
        self.target_fluency = 0.8
        self.min_fluency = 0.5
        self.max_fluency = 1.0

        # Store what the agent tried
        self.curr_step = 0
        self.are_we_done = False

    def set_episode_length(self, day_interval):
        """
        :param day_interval: how often we will record information, make a recommendation
        The smaller this is, the longer an episode (patient trajectory) is
        :return:
        """
        self.day_interval = day_interval
        self.episode_length = 200

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : list of length 1

        Returns
        -------
        observation (state), reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """

        if self.speech_fluency is None:
            raise Exception("You need to reset() the environment before calling step()!")

        # add new action to medication dose list
        self.medication_dose.append(action[0])

        # check if we're done
        if self.curr_step >= self.episode_length - 1:
            self.are_we_done = True

        self._update_state(action[0])

        state = np.array([self.speech_fluency,
                          self.depression,
                          self.anxiety,
                          self.insomnia,
                          self.medication_effect])

        reward = self._get_reward()

        # increment episode
        self.curr_step += 1

        return state, reward, self.are_we_done, {}

    def _update_state(self, medication_dose):
        self.medication_effect = medication_dose / (medication_dose + 1)
        self.speech_fluency = np.clip(self.speech_fluency + self.medication_effect - 0.01 * self.depression, 0, 1)
        self.depression = np.clip(self.depression - self.medication_effect * 0.1, 0, 5)
        self.anxiety = np.clip(self.anxiety - self.medication_effect * 0.1, 0, 5)
        self.insomnia = np.clip(self.insomnia - self.medication_effect * 0.1, 0, 5)

    def _get_reward(self):
        """
        Reward is based on smooth function.
        Target blood glucose level: 80
        g parameter will change slope: 0.7
        """
        reward = 1 - np.tanh(np.abs((self.speech_fluency - self.target_fluency) / 0.1)) ** 2
        if (self.speech_fluency < self.min_fluency) or (self.speech_fluency > self.max_fluency):
            reward = -1.

        return reward

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation (state)
        """

        self.curr_step = 0
        self.are_we_done = False

        # Steady State Initial Conditions for the States
        self.speech_fluency = np.random.uniform(self.min_fluency, self.max_fluency)
        self.depression = np.random.randint(1, 6)
        self.anxiety = np.random.randint(1, 6)
        self.insomnia = np.random.randint(1, 6)
        self.medication_effect = 0.0

        self.medication_dose = [0.0]

        state = np.array([self.speech_fluency,
                          self.depression,
                          self.anxiety,
                          self.insomnia,
                          self.medication_effect])

        return state

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)



if __name__ == "__main__":
    env = SpeechFluencyEnv()
    env.set_episode_length(day_interval=1)
    state = env.reset()
    print("Initial state:", state)

    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        print(f"Step: {env.curr_step}, State: {state}, Reward: {reward}")