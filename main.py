import os
import gym
import gymnasium
from gymnasium import spaces
from gym.utils import seeding
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import pandas as pd

class Medication:
    def __init__(self, name, effect_onset, effect_rate, treatments, available_doses, side_effect_rate, half_life, drug_class):
        self.name = name
        self.time_to_effect = effect_onset
        self.effect_rate = effect_rate
        self.treatments = treatments.split(",")
        # self.available_doses = [float(available_doses)] # for drugs2.csv
        self.available_doses = [float(available_doses) for available_doses in available_doses.split(",")]
        self.side_effect_rate = side_effect_rate
        self.half_life = half_life
        self.drug_class = drug_class

    def __str__(self) -> str:
        return f"Medication: {self.name}\nEffect Onset: {self.time_to_effect}\nEffect Rate: {self.effect_rate}\nTreatments: {self.treatments}\nAvailable Doses: {self.available_doses}\nSide Effect Rate: {self.side_effect_rate}\nHalf Life: {self.half_life}\n"

    def __repr__(self) -> str:
        return f"Medication: {self.name}\nEffect Onset: {self.time_to_effect}\nEffect Rate: {self.effect_rate}\nTreatments: {self.treatments}\nAvailable Doses: {self.available_doses}\nSide Effect Rate: {self.side_effect_rate}\nHalf Life: {self.half_life}\n"
    
    def __eq__(self, o: object) -> bool:
        return self.name == o.name
    
    def __hash__(self) -> int:
        return hash(self.name)

class SpeechFluencyEnv(gymnasium.Env):
    def __init__(self):
        self.__version__ = "0.0.1"

        # General variables defining the environment
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.seed()

        # Lists that will hold episode data
        self.speech_fluency = None
        self.prev_speech_fluency = None
        self.depression = None
        self.anxiety = None
        self.insomnia = None
        self.day = None
        self.side_effect = None
        self.medication_dose = {}

        # df = pd.read_csv('data/drugs2.csv', delimiter='\t')
        df = pd.read_csv('data/modified_drugs.csv', delimiter='\t')
        medications = [Medication(row['name'], row['onset'], row['success_rate'], row['treats'], row['doses'], row['adr_rate'], row['half_life'], row['type']) for _, row in df.iloc[0:15].iterrows()]

        self._medication_dose_pairs = [(medication, dose) for medication in medications for dose in medication.available_doses]

        # Defining action space
        self.action_space = spaces.Discrete(len(self._medication_dose_pairs))

        # Defining observation space (speech fluency)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # # Reward definitions
        # self.target_fluency = 0.8
        # self.min_fluency = 0.0
        # self.max_fluency = 1.0

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
        if self.speech_fluency is None:
            raise Exception("You need to reset() the environment before calling step()!")        

        medication, dose = self._medication_dose_pairs[action]
        if medication in self.medication_dose:
            self.medication_dose[medication] += dose
        else:
            self.medication_dose[medication] = dose
        
        # check if we're done
        if self.curr_step >= self.episode_length - 1:
            self.are_we_done = True

        self._update_state(self._medication_dose_pairs[action])

        state = np.array([self.speech_fluency], dtype=np.float32)

        reward = self._get_reward()

        # increment episode
        self.curr_step += 1

        return state, reward, self.are_we_done, False, {}

    def _update_state(self, medication_dose):
        self.side_effect = 0
        for med, dose in self.medication_dose.items():
            decay = np.exp(-np.log(2) / med.half_life)
            self.medication_dose[med] = dose * decay

            for condition in med.treatments:
                effect = med.effect_rate * dose / 100

                if condition == "depression":
                    self.depression -= effect
                elif condition == "anxiety":
                    self.anxiety -= effect
                elif condition == "insomnia":
                    self.insomnia -= effect

                self.side_effect += med.side_effect_rate * self.medication_dose[med] / 100

        self.depression = np.clip(self.depression, 0, 5)
        self.anxiety = np.clip(self.anxiety, 0, 5)
        self.insomnia = np.clip(self.insomnia, 0, 5)

        self.prev_speech_fluency = self.speech_fluency
        self.speech_fluency = 1 - (self.depression + self.anxiety + self.insomnia) / 15 

    def _get_reward(self):
        reward = (self.speech_fluency - self.prev_speech_fluency) * 100 - self.side_effect
        return reward

    def reset(self, seed=None):
        self.curr_step = 0
        self.are_we_done = False

        # Steady State Initial Conditions for the States
        self.speech_fluency = np.random.uniform(0, 1)
        self.depression = np.random.randint(1, 6)
        self.anxiety = np.random.randint(1, 6)
        self.insomnia = np.random.randint(1, 6)
        self.day = 0
        self.side_effect = 0
        self.medication_dose = {}

        state = np.array([self.speech_fluency], dtype=np.float32)

        return state, {}

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)



if __name__ == "__main__":
    env = SpeechFluencyEnv()
    env.set_episode_length(day_interval=1)
    train = False
    check_env(env)
    
    if train:
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=50000)
        model.save("ppo_speech_fluency")
    else:
        model = PPO.load("ppo_speech_fluency")

    # Evaluate the agent
    state, _ = env.reset()
    done = False
    obj_list = []
    init_obj = {
        "step": 0,
        "speech_fluency": state[0],
        "depression": env.depression,
        "anxiety": env.anxiety,
        "insomnia": env.insomnia,
        "side_effect": env.side_effect,
        "reward": 0,
        "action": None
    }
    obj_list.append(init_obj)
    while not done:
        action, _ = model.predict(state)
        state, reward, done, _, _ = env.step(action)
        print(f"Step: {env.curr_step}, State: {state}, Reward: {reward}")
        print(f"Medication Dose: {action}")
        day_obj = {
            "step": env.curr_step,
            "speech_fluency": state[0],
            "depression": env.depression,
            "anxiety": env.anxiety,
            "insomnia": env.insomnia,
            "side_effect": env.side_effect,
            "reward": reward,
            "action": action
        }
        obj_list.append(day_obj)

    pd.DataFrame(obj_list).to_csv("out/ppo_speech_fluency.csv", index=False)

