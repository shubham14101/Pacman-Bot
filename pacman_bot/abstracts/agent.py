import time
from abc import ABC, abstractmethod

from utility.auxilliary import MultiSearchEvaluationFunctions


class Agent(ABC):
	"""
	This class defines a generic Agent.
	An agent can be `pacman` or the `ghosts`. An agent only provides the next action given the
	current state.
	"""

	def __init__(self, agent_idx = 0):
		self.agent_idx = agent_idx

	def register_initial_state(self, game_state):
		pass

	@abstractmethod
	def get_action(self, game_state):
		"""
		Given a GameState, returns the next action to be taken (direction to move in)
		"""
		raise NotImplementedError()


class MultiAgentSearchAgent(Agent):
	def __init__(self, evaluation_function_name = MultiSearchEvaluationFunctions.ScoreEvaluationFunc.name,
	             depth: str = '2'):
		super(MultiAgentSearchAgent, self).__init__(agent_idx = 0)

		self.evaluation_function = MultiSearchEvaluationFunctions[evaluation_function_name].value
		self.depth = int(depth)

	@abstractmethod
	def get_action(self, game_state):
		pass


class ReinforcementLearningAgent(Agent):
	"""
	Assigns values to (state, action) Q-Values for an environment. As well as a value to a
	state and a policy given respectively by,

	V(s) = max_{a in actions} Q(s,a)
	policy(s) = arg_max_{a in actions} Q(s,a)

	"""

	def __init__(self, action_function = None, alpha = 1.0, epsilon = 0.05, gamma = 0.8, num_training = 10):
		"""
		Sets options, which can be passed in via the Pacman command line using
			alpha    - learning rate
			epsilon  - exploration rate
			gamma    - discount factor
			num_training - number of training episodes, i.e. no learning after these many episodes
		"""

		super().__init__()

		if action_function is None:
			action_function = lambda state: state.get_legal_actions()

		self.action_function = action_function
		self.episodes_so_far = 0
		self.cumulative_train_rewards = 0.0
		self.cumulative_test_rewards = 0.0
		self.num_training = int(num_training)
		self.epsilon = float(epsilon)
		self.alpha = float(alpha)
		self.discount = float(gamma)
		self.last_state = None
		self.last_action = None
		self.episode_rewards = 0.0

		self.last_window_collected_rewards = None
		self.episode_start_time = None

	# To be overridden
	@abstractmethod
	def get_q_value(self, state, action):
		"""
		Should return Q(state,action)
		"""
		raise NotImplementedError()

	@abstractmethod
	def get_value(self, state):
		"""
		V(s) = max_{a in actions} Q(s,a)
		"""
		raise NotImplementedError()

	@abstractmethod
	def get_policy(self, state):
		"""
		policy(s) = arg_max_{a in actions} Q(s,a)
		"""
		raise NotImplementedError()

	@abstractmethod
	def get_action(self, state):
		"""
		state: can call state.get_legal_actions()
		Choose an action and return it.
		"""
		raise NotImplementedError()

	@abstractmethod
	def update(self, state, action, next_state, reward):
		"""
		Called after one step in the game with the reward.
		"""
		raise NotImplementedError()

	def get_legal_actions(self, state):
		"""
		Get the actions available for a given state.
		"""
		return self.action_function(state)

	def observe_transition(self, state, action, next_state, delta_reward):
		"""
		Called by environment to inform agent that a transition has
		been observed.
		"""
		self.episode_rewards += delta_reward
		self.update(state, action, next_state, delta_reward)

	def start_episode(self):
		"""
		Called by environment when new episode is starting
		"""
		self.last_state = None
		self.last_action = None
		self.episode_rewards = 0.0

	def stop_episode(self):
		"""
		Called by environment when episode is done
		"""

		if self.episodes_so_far < self.num_training:
			self.cumulative_train_rewards += self.episode_rewards
		else:
			self.cumulative_test_rewards += self.episode_rewards

		self.episodes_so_far += 1
		if self.episodes_so_far >= self.num_training:
			# Take off the training wheels
			self.epsilon = 0.0  # no exploration
			self.alpha = 0.0  # no learning

	def is_in_training(self):
		return self.episodes_so_far < self.num_training

	def is_in_testing(self):
		return not self.is_in_training()

	###################
	# Pacman Specific #
	###################
	def observation_function(self, state):
		if self.last_state is not None:
			reward = state.get_score() - self.last_state.get_score()
			self.observe_transition(self.last_state, self.last_action, state, reward)
		return state

	def register_initial_state(self, state):
		self.start_episode()
		if self.episodes_so_far == 0:
			print('Beginning %d episodes of Training' % self.num_training)

	def final(self, state):
		"""
		Called by Pacman game at the terminal state
		"""

		delta_reward = state.get_score() - self.last_state.get_score()
		self.observe_transition(self.last_state, self.last_action, state, delta_reward)
		self.stop_episode()

		# Make sure we have this var
		if self.episode_start_time is None:
			self.episode_start_time = time.time()
		if self.last_window_collected_rewards is None:
			self.last_window_collected_rewards = 0.0
		self.last_window_collected_rewards += state.get_score()

		window_size = 100
		if self.episodes_so_far % window_size == 0:
			print('RL Status:')
			window_avg = self.last_window_collected_rewards / float(window_size)
			if self.is_in_training():
				train_avg = self.cumulative_train_rewards / float(self.episodes_so_far)
				print('\tCompleted %d out of %d training episodes' % (self.episodes_so_far, self.num_training))
				print('\tAverage Rewards over all training: %.2f' % train_avg)
			else:
				test_avg = float(self.cumulative_test_rewards) / (self.episodes_so_far - self.num_training)
				print('\tCompleted %d test episodes' % (self.episodes_so_far - self.num_training))
				print('\tAverage Rewards over testing: %.2f' % test_avg)
			print('\tAverage Rewards for last %d episodes: %.2f' % (window_size, window_avg))
			print('\tEpisode took %.2f seconds' % (time.time() - self.episode_start_time))

			self.last_window_collected_rewards = 0.0
			self.episode_start_time = time.time()

		if self.episodes_so_far == self.num_training:
			print('Training Done')
