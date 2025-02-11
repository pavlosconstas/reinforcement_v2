import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import matplotlib.pyplot as plt
import pandas as pd

from hmmlearn import hmm

class Patient:
    def __init__(self):
        self.depression = np.random.randint(1, 6)
        self.anxiety = np.random.randint(1, 6)
        self.insomnia = np.random.randint(1, 6)

        self.speech_fluency = np.clip(np.random.normal(0.5, 0.1), 0, 1)

        self.medication_accumulation = {}
        self.side_effects = {}

        self.day = 0

        self.depression_model = self.generate_model(self.depression)
        self.anxiety_model = self.generate_model(self.anxiety)
        self.insomnia_model = self.generate_model(self.insomnia)

        self.hidden_states = np.array([self.depression, self.anxiety, self.insomnia])
        self.score = self.calculate_speech_fluency()
        self.previous_fluency = self.score

    def calculate_speech_fluency(self):
        depression_score = (6 - self.depression) / 5
        anxiety_score = (6 - self.anxiety) / 5
        insomnia_score = (6 - self.insomnia) / 5

        speech_fluency = (0.25 * depression_score + 0.25 * anxiety_score + 0.25 * insomnia_score + 0.25 * self.speech_fluency)
        return speech_fluency

    def generate_observation(self):
        return np.array([self.speech_fluency, 
                         self.depression_model.sample(1)[0].mean(), 
                         self.anxiety_model.sample(1)[0].mean(), 
                         self.insomnia_model.sample(1)[0].mean()], dtype=np.float32)

    def take_medication(self, medication, dosage: float):
        if medication in self.medication_accumulation:
            self.medication_accumulation[medication] += dosage
        else:
            self.medication_accumulation[medication] = dosage
            medication.day_administered = self.day

    def generate_model(self, score: int):
        model = hmm.GaussianHMM(n_components=5, covariance_type="full", init_params="") 
        model.startprob_ = np.array([0.6, 0.3, 0.1, 0.0, 0.0])
        model.transmat_ = np.array([[0.7, 0.2, 0.1, 0.0, 0.0],
                                    [0.0, 0.7, 0.2, 0.1, 0.0],
                                    [0.0, 0.0, 0.7, 0.2, 0.1],
                                    [0.0, 0.0, 0.0, 0.7, 0.3],
                                    [0.0, 0.0, 0.0, 0.0, 1.0]])
        
        # this just random data i dont know what to put here
        X = np.random.normal(loc=score, scale=1, size=(100, 1))
        model.means_ = np.array([[score - 1], [score - 0.5], [score], [score + 0.5], [score + 1]])
        model.covars_ = np.tile(np.identity(1), (5, 1, 1)) * 0.5

        model.fit(X)
        return model

    
    def adjust_transition_matrix(self, matrix, effect):
        transition_matrix = matrix.copy()

        for i in range(len(transition_matrix)):
            for j in range(len(transition_matrix[i])):
                transition_matrix[i, j] *= (1 + effect)

        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
        return transition_matrix
    
    def compute_reward(self):
        return self.calculate_speech_fluency() - self.previous_fluency
    
    def update(self):
        for med, dose in self.medication_accumulation.items():
            decay_factor = np.exp(-np.log(2) / med.half_life)
            self.medication_accumulation[med] = dose * decay_factor

            if med.check_if_is_active(self.day):
                for condition in med.treatments:
                    EC50 = np.mean(med.available_doses)
                    effect = np.random.normal(med.effect_rate, 0.1) * dose / (dose + EC50)

                    if condition == "depression":
                        self.depression_model.transmat_ = self.adjust_transition_matrix(self.depression_model.transmat_, effect)
                    elif condition == "anxiety":
                        self.anxiety_model.transmat_ = self.adjust_transition_matrix(self.anxiety_model.transmat_, effect)
                    elif condition == "insomnia":
                        self.insomnia_model.transmat_ = self.adjust_transition_matrix(self.insomnia_model.transmat_, effect)

                    current_value = getattr(self, condition)
                    setattr(self, condition, max(1, min(5, current_value - effect)))

        self.hidden_states = np.array([self.depression, self.anxiety, self.insomnia])

        if self.day % 50 == 0:
            print(f"Day: {self.day}, Speech Fluency: {self.score}")

        self.day += 1
        self.previous_fluency = self.score
        self.score = self.calculate_speech_fluency()
    
    def get_true_hidden_states(self):
        return self.hidden_states


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
        self.day_administered = 0
        self.is_active = False
    
    def check_if_is_active(self, current_day):
        self.is_active = current_day >= self.day_administered + self.time_to_effect
        return self.is_active
    
    def __str__(self) -> str:
        return f"Medication: {self.name}\nEffect Onset: {self.time_to_effect}\nEffect Rate: {self.effect_rate}\nTreatments: {self.treatments}\nAvailable Doses: {self.available_doses}\nSide Effect Rate: {self.side_effect_rate}\nHalf Life: {self.half_life}\n"

    def __repr__(self) -> str:
        return f"Medication: {self.name}\nEffect Onset: {self.time_to_effect}\nEffect Rate: {self.effect_rate}\nTreatments: {self.treatments}\nAvailable Doses: {self.available_doses}\nSide Effect Rate: {self.side_effect_rate}\nHalf Life: {self.half_life}\n"
    
    def __eq__(self, o: object) -> bool:
        return self.name == o.name
    
    def __hash__(self) -> int:
        return hash(self.name)

class MedicationEnvironment(py_environment.PyEnvironment):
    def __init__(self, patient: Patient, medications: list[Medication]):
        self._patient = patient
        self._medications = medications
        self._medication_dose_pairs = [(medication, dose) for medication in medications for dose in medication.available_doses]
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(self._medication_dose_pairs)-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,), dtype=np.float32, minimum=0, maximum=1, name='observation')
        self._time_step_spec = ts.time_step_spec(self._observation_spec)
        self._state = np.zeros(4, dtype=np.float32)
        self._reward = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _update_state(self):
        self._state = self._patient.generate_observation()

    def _reset(self):
        self._patient = Patient()
        self._update_state()
        self._episode_ended = False
        self._reward = 0
        self._state = np.zeros(4, dtype=np.float32)
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self._reset()

        medication, dose = self._medication_dose_pairs[action]
        self._patient.take_medication(medication, dose)
        self._patient.update()
        self._update_state()

        if self._patient.day == 200:
            self._episode_ended = True
            return ts.termination(self._state, self._patient.compute_reward())

        return ts.transition(self._state, reward=self._patient.compute_reward(), discount=1.0)

    def get_true_hidden_states(self):
        return self._patient.get_true_hidden_states()

def compute_avg_return(environment, policy, num_eval_episodes=10):
    total_return = 0.0
    for _ in range(num_eval_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_eval_episodes
    return avg_return

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)

if __name__ == "__main__":
    patient = Patient()
    df = pd.read_csv('data/drugs.csv', delimiter='\t')
    medications = [Medication(row['name'], row['onset'], row['success_rate'], row['treats'], row['doses'], row['adr_rate'], row['half_life'], row['type']) for _, row in df.iloc[0:15].iterrows()]

    env = MedicationEnvironment(patient, medications)
    tf_env = tf_py_environment.TFPyEnvironment(env)

    agent = lin_ucb_agent.LinearUCBAgent(
        time_step_spec=tf_env.time_step_spec(),
        action_spec=tf_env.action_spec(),
        tikhonov_weight=1.0,
        alpha=10.0,
        dtype=tf.float32)

    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=1000)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        env=tf_env,
        policy=agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=20)

    collect_driver.run()

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=64,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)
    trajectories, _ = next(iterator)

    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)

    avg_return = compute_avg_return(tf_env, agent.policy, num_eval_episodes=1)
    returns = [avg_return]
    iterations = [0]

    num_iterations = 100
    collect_steps_per_iteration = 10
    replay_buffer_max_length = 1000
    log_interval = 10
    eval_interval = 10

    true_hidden_states = []
    estimated_hidden_states = []

    for i in range(num_iterations):
        for _ in range(collect_steps_per_iteration):
            collect_step(tf_env, agent.collect_policy, replay_buffer)

        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        true_hidden_states.append(env.get_true_hidden_states())
        estimated_hidden_states.append(tf_env.current_time_step().observation.numpy().flatten())

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(tf_env, agent.policy, num_eval_episodes=1)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)
            iterations.append(step)

        step += 1

    returns = np.array(returns).flatten()

    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=5)
    plt.show()

    true_hidden_states = np.array(true_hidden_states)
    estimated_hidden_states = np.array(estimated_hidden_states)

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    axs[0].plot(true_hidden_states[:, 0], label='True Depression')
    axs[0].plot(estimated_hidden_states[:, 1], label='Estimated Depression')
    axs[0].legend()
    axs[0].set_title('Depression')

    axs[1].plot(true_hidden_states[:, 1], label='True Anxiety')
    axs[1].plot(estimated_hidden_states[:, 2], label='Estimated Anxiety')
    axs[1].legend()
    axs[1].set_title('Anxiety')

    axs[2].plot(true_hidden_states[:, 2], label='True Insomnia')
    axs[2].plot(estimated_hidden_states[:, 3], label='Estimated Insomnia')
    axs[2].legend()
    axs[2].set_title('Insomnia')

    plt.show()
