import random

import numpy as np

from abstracts.agent import Agent, MultiAgentSearchAgent, ReinforcementLearningAgent
from algorithms import Algorithms, AntColonyOptimization, SearchHeuristics
from algorithms.alphabetapruning import alphabeta
from algorithms.minimax import minimax
from pacman.game import Actions, Ghost
from pacman.searchproblems import PositionalSearchProblem
from utility.auxilliary import Direction, load_dict, manhattan_distance, save_dict


# ### Pacman Agents ###
class PositionalSearchAgent(Agent):
	"""
	Searches for a food particle.
	Simple search from point A to point B.
	"""

	def __init__(self, search_function_name = Algorithms.DepthFirstSearch.name,
	             heuristic_name = SearchHeuristics.NullHeuristic.name):
		super(PositionalSearchAgent, self).__init__(agent_idx = 0)

		search_function = Algorithms[search_function_name].value
		heuristic = SearchHeuristics[heuristic_name].value

		self.search_function = lambda x: search_function(x, heuristic = heuristic)
		self.actions = None
		self.action_index = -1

	def register_initial_state(self, state):

		if self.search_function is None:
			raise Exception('No search function!')

		problem = PositionalSearchProblem(state)
		self.actions = self.search_function(problem)

	def get_action(self, state):

		if self.action_index < 0:
			self.action_index = 0

		i = self.action_index
		self.action_index += 1
		if i < len(self.actions):
			return self.actions[i]
		else:
			return Direction.STOP


class FoodSearchAgent(Agent):
	"""
    This search problem can be used to find a path such that all food particles are consumed starting from the
    initial position in approx min cost possible.

    The state space consists of (m,n) positions in a pacman game.
    TSP search agent.
    """

	def __init__(self, search_function_name = Algorithms.DepthFirstSearch.name,
	             heuristic_name = SearchHeuristics.NullHeuristic.name, ant_count = '1000', alpha = '0.1', beta = '10',
	             pheromone_evaporation_coeff = '0.1', pheromone_deposition_constant = '1000', max_iterations = '10000'):
		super(FoodSearchAgent, self).__init__()

		search_function = Algorithms[search_function_name].value
		heuristic = SearchHeuristics[heuristic_name].value

		self.search_function = lambda x: search_function(x, heuristic = heuristic)
		self.actions = None
		self.action_index = -1

		self.ant_count = int(ant_count)
		self.alpha = float(alpha)
		self.beta = float(beta)
		self.pheromone_evaporation_coeff = float(pheromone_evaporation_coeff)
		self.pheromone_deposition_constant = float(pheromone_deposition_constant)
		self.max_iterations = int(max_iterations)

	def register_initial_state(self, game_state):
		self.search(game_state)

	def search(self, state):
		pacman_pos = None
		for pacman, pos in state.layout.agent_positions:
			if pacman:
				pacman_pos = pos
				break

		if pacman_pos is None:
			return None

		# map to store all paths
		meta_map = dict()

		food_positions = state.layout.total_food

		meta_map[pacman_pos] = list()
		for pos in food_positions:
			meta_map[pos] = list()

		# paths from pacman to all food particles
		for pos in food_positions:
			problem = PositionalSearchProblem(state, pacman_pos, pos)
			solution = self.search_function(problem)
			meta_map[pacman_pos].append((pos, solution, len(solution)))

		# paths between all food particles
		for idx1, pos1 in enumerate(food_positions):
			for idx2, pos2 in enumerate(food_positions):
				if idx1 == idx2:
					continue
				problem = PositionalSearchProblem(state, pos1, pos2)
				solution = self.search_function(problem)
				meta_map[pos1].append((pos2, solution, len(solution)))

		aco = AntColonyOptimization(
			model = meta_map,
			start = pacman_pos,
			ant_count = self.ant_count,
			alpha = self.alpha,
			beta = self.beta,
			pheromone_evaporation_coeff = self.pheromone_evaporation_coeff,
			pheromone_deposition_constant = self.pheromone_deposition_constant,
			max_iterations = self.max_iterations
		)
		tour, tour_cost = aco.run_ant_colony_optimization()

		action_path = list()
		for idx in range(len(tour) - 1):

			pos_current = (tour[idx].mlayout, tour[idx].nlayout)
			pos_next = (tour[idx + 1].mlayout, tour[idx + 1].nlayout)

			found = False
			for tup in meta_map[pos_current]:
				if tup[0] == pos_next:
					found = True
					action_path.extend(tup[1])
					break

			if found:
				continue

			for tup in meta_map[pos_next]:
				if tup[0] == pos_current:
					found = True
					action_path.extend(tup[1])
					break

			if not found:
				return None

		self.actions = action_path
		return action_path, tour_cost

	def get_action(self, game_state):
		if self.action_index < 0:
			self.action_index = 0

		i = self.action_index
		self.action_index += 1
		if i < len(self.actions):
			return self.actions[i]
		else:
			return Direction.STOP


class MinimaxAgent(MultiAgentSearchAgent):
	"""
	Agent using MiniMax to select its next move
	"""

	def get_action(self, game_state):
		"""
		Uses the MiniMax algorithm to select the next action.
		"""

		evaluation = minimax(game_state, self.agent_idx, self.evaluation_function, 0, self.depth)
		action = evaluation[1]
		return action


class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	Agent using MiniMax to select its next move
	"""

	def get_action(self, game_state):
		"""
		Uses the MiniMax algorithm to select the next action.
		"""

		evaluation = alphabeta(game_state, self.agent_idx, self.evaluation_function, -float("inf"), float("inf"), 0,
		                       self.depth)
		action = evaluation[1]
		return action


class ReflexAgent(Agent):
	"""
	Acts on immediate surroundings.
	"""

	def __init__(self, *args, **kwargs):
		super(ReflexAgent, self).__init__()

	def get_action(self, game_state):
		legal_moves = game_state.get_legal_actions()
		scores = np.array([])
		for action in legal_moves:
			scores = np.append(scores, self.evaluation_function(game_state, action))
		best_score = np.max(scores)
		max_index = np.where(scores == best_score)
		index = np.random.choice(max_index.shape[0], 1, replace = False)
		best_index = max_index[index]
		return legal_moves[best_index]

	def evaluation_function(self, game_state, action):
		successor_state = game_state.generate_pacman_successor(action)
		new_position = successor_state.get_pacman_position()
		food = successor_state.get_food()
		ghost_states = successor_state.get_ghost_states()

		for state in ghost_states:
			distance_ghost = manhattan_distance(new_position, state.get_position())
			if distance_ghost <= 1:
				return -1e7
		food_position = food.as_list()
		shortest_food = 1e7
		count = 0
		next_food = successor_state.get_pacman_position()
		for pos in food_position:
			food_distance = manhattan_distance(new_position, pos)
			if food_distance < shortest_food:
				shortest_food = food_distance
				next_food = pos
			count = count + 1
		if count == 0:
			shortest_food = 0

		score = -shortest_food
		distance_f = 1e7
		for state in ghost_states:
			distance_ghost_food = manhattan_distance(next_food, state.get_position())
			if distance_ghost_food < distance_f:
				distance_f = distance_ghost_food
		score = score + distance_f
		return score


class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	Expectimax Agent
	"""

	def get_action(self, game_state):
		root = self.value(game_state, 0, self.agent_idx)
		action = root[1]
		return action

	def value(self, game_state, current_depth, agent_idx):
		if agent_idx == game_state.get_num_agents():
			current_depth = current_depth + 1
			agent_idx = 0
		legal_action = game_state.get_legal_actions(agent_idx)
		if len(legal_action) == 0:
			return self.evaluation_function(game_state)
		if current_depth == self.depth:
			return self.evaluation_function(game_state)
		if agent_idx == 0:
			return self.max(game_state, current_depth, agent_idx)
		else:
			return self.min(game_state, current_depth, agent_idx)

	def max(self, game_state, current_depth, agent_idx):
		node = [-1e7]
		action_list = game_state.get_legal_actions(agent_idx)
		for action in action_list:
			successor = game_state.generate_successor(agent_idx, action)
			successor_val = self.value(successor, current_depth, agent_idx + 1)
			if successor_val[0] >= node[0]:
				node = successor_val[0], action

		return node

	def min(self, game_state, current_depth, agent_idx):
		node = [1e7]
		action_list = game_state.get_legal_actions(agent_idx)
		for action in action_list:
			successor = game_state.generate_successor(agent_idx, action)
			successor_val = self.value(successor, current_depth, agent_idx + 1)
			if successor_val[0] <= node[0]:
				node = successor_val[0], action

		return node


class QLearningAgent(ReinforcementLearningAgent):
	"""
	Q-Learning Agent
	"""

	def __init__(self, **args):

		self.save_model_file = args.pop('save')
		self.load_model_file = args.pop('load')

		ReinforcementLearningAgent.__init__(self, **args)

		if self.load_model_file is not None:
			self._q = load_dict(self.load_model_file)
		else:
			self._q = dict()

	def get_q_value(self, state, action):
		"""
		Returns Q(state,action)
		Should return 0.0 if we have never seen a state
		or the Q node value otherwise
		"""

		key = (state, action)

		try:
			q_value = self._q[key]
		except KeyError:
			q_value = self._q[key] = 0.0
		return q_value

	def get_action(self, state):
		"""
		Epsilon greedy if training
		"""

		# epsilon-greedy in training
		if self.is_in_training():
			r = np.random.rand()
			if r < self.epsilon:
				# choose random action
				return np.random.choice(self.get_legal_actions(state))

		# return action from policy
		action = self.get_policy(state)

		self.last_state = state
		self.last_action = action

		return action

	def update(self, state, action, next_state, reward):
		"""
		Update q-value
		"""

		rsa = reward
		qsa = self.get_q_value(state, action)

		if np.isinf(rsa):
			new_q = rsa
		elif np.isinf(qsa):
			new_q = qsa
		else:
			new_q = qsa + self.alpha * (rsa + (self.discount * self.get_value(next_state)) - qsa)

		key = (state, action)
		self._q[key] = new_q

	# print('Updated Q-value', qsa, new_q)

	def get_policy(self, state):
		legal_actions = self.get_legal_actions(state)

		if len(legal_actions) == 0:
			return None

		values = list()
		for action in legal_actions:
			q = self.get_q_value(state, action)
			values.append((action, q))

		max_q = values[0]
		for a, q in values:
			if q > max_q[1]:
				max_q = (a, q)
		return max_q[0]

	def get_value(self, state):
		legal_actions = self.get_legal_actions(state)

		if len(legal_actions) == 0:
			return 0.0

		values = list()
		for action in legal_actions:
			values.append(self.get_q_value(state, action))
		return max(values)

	def final(self, state):
		super().final(state)

		# played last training game, save model
		if self.episodes_so_far == self.num_training:
			if self.save_model_file is not None:
				save_dict(self._q, self.save_model_file)


# ### Ghost Agents ###
class RandomAgent(Agent):
	"""A ghost that chooses a legal action uniformly at random."""

	def get_action(self, state):
		# Collect legal moves and successor states
		legal_moves = Ghost.get_legal_actions(state, self.agent_idx)
		# Choose one of the legal actions
		return random.choice(legal_moves)  # Pick randomly among the legal


class ChasingGhostAgent(Agent):
	def get_action(self, game_state):
		# Read variables from state
		ghost_state = game_state.get_ghost_state(self.agent_idx)
		legal_actions = game_state.get_legal_actions(self.agent_idx)
		pos = game_state.get_ghost_position(self.agent_idx)
		is_scared = ghost_state.scared_timer > 0

		action_vectors = [Actions.direction_to_vector(a) for a in legal_actions]
		new_positions = [(pos[0] + a[0], pos[1] + a[1]) for a in action_vectors]
		pacman_position = game_state.get_pacman_position()

		# Select best actions given the state
		distances_to_pacman = [manhattan_distance(pos, pacman_position) for pos in new_positions]
		if is_scared:
			best_score = max(distances_to_pacman)
			best_prob = self.prob_scared
		else:
			best_score = min(distances_to_pacman)
			best_prob = self.prob_attack
		best_actions = [action for action, distance in zip(legal_actions, distances_to_pacman) if
		                distance == best_score]

		# choose action
		selection_prob = [0] * len(legal_actions)
		for idx, action in enumerate(legal_actions):
			if action in best_actions:
				selection_prob[idx] += best_prob / len(best_actions)
			selection_prob[idx] += (1 - best_prob) / len(legal_actions)

		action = np.random.choice(legal_actions, p = selection_prob)
		return action

	def __init__(self, agent_idx, prob_attack = 0.8, prob_scared = 0.8):
		super().__init__(agent_idx)

		self.agent_idx = agent_idx
		self.prob_attack = prob_attack
		self.prob_scared = prob_scared
