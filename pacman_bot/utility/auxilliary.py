import enum
from functools import partial

import numpy

from .error import raise_unexpected_behaviour
from six.moves import cPickle as pickle


class Matrix:
	"""
	A 2-Dimensional matrix Data Structure created using a list of lists.
	The index notation refers to the matrix co-ordinate layout i.e., m = row and n = col.
	"""

	def __init__(self, height: int, width: int, data_type, init_value = 0):

		if data_type not in [int, float, str, bool]:
			raise ValueError('`data_type` can only be `int`, `float`, `bool` or `str`!')

		self.data_type = data_type
		self.init_value = self.data_type(init_value)

		self.height = height
		self.width = width
		self.data = [
			[self.init_value for n in range(self.width)]
			for m in range(self.height)
		]

	def __getitem__(self, i):
		return self.data[i]

	def __setitem__(self, key, item):
		self.data[key] = item

	def __str__(self):
		out = [
			[str(self.data[m][n]) for n in range(self.width)]
			for m in range(self.height)
		]
		return '\n'.join([''.join(x) for x in out])

	def __eq__(self, other):
		if other is None:
			return False
		return self.data == other.data

	def copy(self):
		g = Matrix(self.height, self.width, self.data_type)
		g.data = [x[:] for x in self.data]
		return g

	def deep_copy(self):
		return self.copy()

	def shallow_copy(self):
		g = Matrix(self.height, self.width, self.data_type)
		g.data = self.data
		return g

	def as_list(self):
		out_list = []
		for x in self.data:
			out_list.extend(x)
		return out_list


class BooleanMatrix(Matrix):
	""" Stores only booleans """

	def __init__(self, height: int, width: int, init_value = False):
		super(BooleanMatrix, self).__init__(height, width, bool, init_value)

	def __hash__(self):
		base = 1
		h = 0
		for l in self.data:
			for i in l:
				if i:
					h += base
				base *= 2
		return hash(h)

	def copy(self):
		g = BooleanMatrix(self.height, self.width)
		g.data = [x[:] for x in self.data]
		return g

	def deep_copy(self):
		return self.copy()

	def shallow_copy(self):
		g = BooleanMatrix(self.height, self.width)
		g.data = self.data
		return g

	def count(self, item = True):
		return sum([x.count(item) for x in self.data])

	def as_list(self, key = True):
		out_list = []
		for m in range(self.height):
			for n in range(self.width):
				if self[m][n] == key:
					out_list.append((m, n))
		return out_list


class Direction:
	NORTH = 'North'
	SOUTH = 'South'
	EAST = 'East'
	WEST = 'West'
	STOP = 'Stop'

	LEFT = {
		NORTH: WEST,
		SOUTH: EAST,
		EAST: NORTH,
		WEST: SOUTH,
		STOP: STOP
	}

	RIGHT = dict([(y, x) for x, y in LEFT.items()])

	REVERSE = {
		NORTH: SOUTH,
		SOUTH: NORTH,
		EAST: WEST,
		WEST: EAST,
		STOP: STOP
	}


class Actions:
	"""
	A collection of static methods for manipulating move actions.
	Directions are just mapped to unit vectors and then simple math.
	Again in terms of Matrix co-ordinates [m, n] not cartesian [x, y].
	"""

	# Directions
	_directions = {
		Direction.NORTH: (-1, 0),
		Direction.SOUTH: (1, 0),
		Direction.EAST: (0, 1),
		Direction.WEST: (0, -1),
		Direction.STOP: (0, 0)
	}

	_directions_as_list = _directions.items()

	TOLERANCE = .001

	@staticmethod
	def reverse_direction(action):
		if action == Direction.NORTH:
			return Direction.SOUTH
		if action == Direction.SOUTH:
			return Direction.NORTH
		if action == Direction.EAST:
			return Direction.WEST
		if action == Direction.WEST:
			return Direction.EAST
		return action

	@staticmethod
	def vector_to_direction(vector):
		dm, dn = vector
		if dm < 0:
			return Direction.NORTH
		if dm > 0:
			return Direction.SOUTH
		if dn < 0:
			return Direction.WEST
		if dn > 0:
			return Direction.EAST
		return Direction.STOP

	@staticmethod
	def direction_to_vector(direction):
		dm, dn = Actions._directions[direction]
		return dm, dn

	@staticmethod
	def get_possible_actions(config, walls):
		possible = []
		m, n = config.position

		# # In between grid points, all agents must continue straight
		# if abs(x - x_int) + abs(y - y_int) > Actions.TOLERANCE:
		# 	return [config.getDirection()]

		for direction, vector in Actions._directions_as_list:
			dm, dn = vector
			next_m = m + dm
			next_n = n + dn
			if not walls[next_m][next_n]:
				possible.append(direction)

		return possible

	@staticmethod
	def get_legal_neighbors(position, walls):
		m, n = position

		neighbors = []
		for direction, vector in Actions._directions_as_list:
			dm, dn = vector
			next_m = m + dm
			if next_m < 0 or next_m >= walls.height:
				continue
			next_n = n + dn
			if next_n < 0 or next_n >= walls.width:
				continue
			if not walls[next_m][next_n]:
				neighbors.append((next_m, next_n))
		return neighbors

	@staticmethod
	def get_successor(position, action):
		dm, dn = Actions.direction_to_vector(action)
		m, n = position
		return m + dm, n + dn


def traverse_path(node, meta_map):
	"""
	Forms an ordered list of actions to perform to reach from a start state to goal node
	:param node: goal node
	:param meta_map: map of state -> (parent state, action, ...)
	:return: list of actions
	"""

	action_list = list()

	while True:

		mapping = meta_map[node]

		if len(mapping) < 2:
			raise_unexpected_behaviour()
			break

		if (mapping[0] is not None) and (mapping[1] is not None):
			# parent state
			node = mapping[0]
			# action taken to reach from parent state to child state
			action = mapping[1]

			action_list.append(action)
		else:
			# starting node reached
			break

	action_list.reverse()
	return action_list


def euclidean_distance(pos1: tuple, pos2: tuple):
	"""
	Returns euclidean distance between two co-ordinates
	NOTE: euclidean distance does not differ for cartesian v/s matrix layout
	"""

	m1, n1 = pos1
	m2, n2 = pos2

	a = m1 - m2
	b = n1 - n2
	sq_sum = numpy.power(a, 2) + numpy.power(b, 2)
	return numpy.sqrt(sq_sum)


def manhattan_distance(pos1: tuple, pos2: tuple):
	"""
	Returns euclidean distance between two co-ordinates
	NOTE: manhattan distance does not differ for cartesian v/s matrix layout
	"""

	m1, n1 = pos1
	m2, n2 = pos2

	a = m1 - m2
	b = n1 - n2
	abs_sum = numpy.abs(a) + numpy.abs(b)
	return abs_sum


def score_evaluation_function(game_state):
	return game_state.get_score()


def food_count_evaluation_function(game_state):
	# ### WON ###
	# if all the food is over, we won -> maximum evaluated value
	food_count = game_state.get_num_food()
	if food_count == 0:
		return float("inf")

	# ### LOST ###
	ghost_states = game_state.get_ghost_states()
	pacman_pos = game_state.get_pacman_position()
	for state in ghost_states:
		dist_from_ghost = manhattan_distance(pacman_pos, state.get_position())

		# if this state leads to pacman being killed, we lost -> minimum evaluated value
		if dist_from_ghost <= 1:
			return float("-inf")

	# ### ELSE ###
	# calculate based on closest food
	food = game_state.get_food()
	food_available = []
	food_data = []
	for m in range(food.height):
		for n in range(food.width):
			if food[m][n]:
				food_location = (m, n)
				food_available.append(food_location)

	for food_loc in food_available:
		food_distance = manhattan_distance(pacman_pos, food_loc)
		food_data.append(food_distance)

	closest_food_dist = min(food_data)

	# calculate score
	score = (5 * (-closest_food_dist) + (200 * (-food_count)))
	return score


class MultiSearchEvaluationFunctions(enum.Enum):
	ScoreEvaluationFunc = partial(score_evaluation_function)
	FoodCountEvaluationFunc = partial(food_count_evaluation_function)


def save_dict(di_, filename_):
	with open(filename_, 'wb') as f:
		pickle.dump(di_, f)


def load_dict(filename_):
	with open(filename_, 'rb') as f:
		ret_di = pickle.load(f)
	return ret_di
