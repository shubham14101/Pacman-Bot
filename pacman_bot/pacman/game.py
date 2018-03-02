import traceback
from collections import namedtuple

from pacman.layout import Layout, LayoutCharacters
from utility.auxilliary import Actions, Direction, Matrix, manhattan_distance

TIME_PENALTY = 1
SCARED_TIME = 40
COLLISION_TOLERANCE = 0.7


class ClassicGameRules:
	def __init__(self):
		self.initial_state = None

	def new_game(self, layout: Layout, pacman, ghosts, display, quiet):
		agents = [pacman] + ghosts[:layout.get_num_ghosts()]

		initial_state = GameState()
		initial_state.initialize(layout, len(ghosts))

		game = Game(agents, display, self)
		game.state = initial_state

		self.initial_state = initial_state.deep_copy()
		return game

	@staticmethod
	def process(game_state, game):
		if game_state.is_win():
			print("Pacman emerges victorious! Score: %d" % game_state.score)
			game.game_over = True

		if game_state.is_lose():
			print("Pacman died! Score: %d" % game_state.score)
			game.game_over = True

	def get_progress(self, game):
		return float(game.state.get_num_food()) / self.initial_state.get_num_food()


class Pacman:
	"""
	These functions govern how pacman interacts with its environment under
	the classic game rules.
	"""

	@staticmethod
	def get_legal_actions(game_state):
		return Actions.get_possible_actions(game_state.get_pacman_state().configuration, game_state.layout.walls)

	@staticmethod
	def apply_action(game_state, action):

		legal = Pacman.get_legal_actions(game_state)
		if action not in legal:
			raise Exception("Illegal action " + str(action))

		pacman_state = game_state.agent_states[0]

		# Update Configuration
		vector = Actions.direction_to_vector(action)
		pacman_state.configuration = pacman_state.generate_successor(pacman_state.configuration, vector)

		# Eat
		next = pacman_state.configuration.position
		Pacman.consume(next, game_state)

	@staticmethod
	def consume(position, state):
		m, n = position

		# Eat food
		if state.food[m][n]:
			state.score_change += 10
			state.food = state.food.copy()
			state.food[m][n] = False
			state._food_eaten = position

			num_food = state.get_num_food()
			if num_food == 0 and not state._lose:
				state.score_change += 500
				state._win = True

		# Eat capsule
		if position in state.get_capsules():
			state.capsules.remove(position)
			state._capsule_eaten = position

			# Reset all ghosts' scared timers
			for index in range(1, len(state.agent_states)):
				state.agent_states[index].scared_timer = SCARED_TIME


class Ghost:
	@staticmethod
	def get_legal_actions(game_state, ghost_index):
		conf = game_state.get_ghost_state(ghost_index).configuration
		possible_actions = Actions.get_possible_actions(conf, game_state.layout.walls)
		reverse = Actions.reverse_direction(conf.direction)

		if Direction.STOP in possible_actions:
			possible_actions.remove(Direction.STOP)
		if reverse in possible_actions and len(possible_actions) > 1:
			possible_actions.remove(reverse)
		return possible_actions

	@staticmethod
	def apply_action(game_state, action, ghost_index):
		legal = Ghost.get_legal_actions(game_state, ghost_index)
		if action not in legal:
			raise Exception("Illegal ghost action " + str(action))

		ghost_state = game_state.agent_states[ghost_index]
		speed = 1.0
		if ghost_state.scaredTimer > 0:
			speed /= 2.0
		vector = Actions.direction_to_vector(action)
		ghost_state.configuration = ghost_state.generate_successor(ghost_state.configuration, vector)

	@staticmethod
	def decrement_timer(ghost_state):
		timer = ghost_state.scared_timer
		ghost_state.scared_timer = max(0, timer - 1)

	@staticmethod
	def check_death(game_state, agent_index):
		pacman_position = game_state.get_pacman_position()
		if agent_index == 0:
			for index in range(1, len(game_state.agent_states)):
				ghost_state = game_state.agent_states[index]
				ghost_position = ghost_state.get_position()
				if Ghost.can_kill(pacman_position, ghost_position):
					Ghost.collide(game_state, ghost_state, index)
		else:
			ghost_state = game_state.agent_states[agent_index]
			ghost_position = ghost_state.get_position()
			if Ghost.can_kill(pacman_position, ghost_position):
				Ghost.collide(game_state, ghost_state, agent_index)

	@staticmethod
	def collide(game_state, ghost_state, agent_index):
		if ghost_state.scared_timer > 0:
			game_state.score_change += 200
			Ghost.place_ghost(ghost_state)
			ghost_state.scaredTimer = 0
			# Added for first-person
			game_state.killed[agent_index] = True
		else:
			if not game_state._win:
				game_state.score_change -= 500
				game_state._lose = True

	@staticmethod
	def can_kill(pacman_position, ghost_position):
		return manhattan_distance(ghost_position, pacman_position) <= COLLISION_TOLERANCE

	@staticmethod
	def place_ghost(ghost_state):
		ghost_state.configuration = ghost_state.start_configuration


class AgentState:
	"""
	Holds state of an agent.
	A configuration refers to a tuple => (position, direction)
	"""

	def __init__(self, position: tuple, direction: Direction, is_pacman: bool = False):
		self.start_configuration = Configuration(position, direction)
		self.configuration = self.start_configuration
		self.is_pacman = is_pacman
		self.scared_timer = 0
		self.num_carry = 0
		self.num_return = 0

	def __str__(self):
		if self.is_pacman:
			return "Pacman: " + str(self.configuration)
		else:
			return "Ghost: " + str(self.configuration)

	def __eq__(self, other):
		if other is None:
			return False
		return self.configuration == other.configuration and self.scared_timer == other.scared_timer

	def __hash__(self):
		return hash(hash(self.configuration) + 13 * hash(self.scared_timer))

	def copy(self):
		clone = AgentState(self.start_configuration[0], self.start_configuration[1], self.is_pacman)
		clone.configuration = self.configuration
		clone.scaredTimer = self.scared_timer
		clone.numCarrying = self.num_carry
		clone.numReturned = self.num_return
		return clone

	def get_position(self):
		if self.configuration is None:
			return None
		return self.configuration.position

	def get_direction(self):
		return self.configuration.direction

	@staticmethod
	def generate_successor(configuration, vector):
		m, n = configuration.position
		dm, dn = vector
		direction = Actions.vector_to_direction(vector)
		if direction == Direction.STOP:
			direction = configuration.direction
		return Configuration((m + dm, n + dn), direction)


Configuration = namedtuple('Configuration', ['position', 'direction'])


class GameState:
	"""
	Holds the information about the state of the Game at a given time.
	Crucial to the running of the game.
	"""

	def __init__(self, prev_state = None):
		# copy data from previous state if provided
		if prev_state is not None:
			self.layout = prev_state.layout
			self.food = prev_state.food.shallow_copy()
			self.capsules = prev_state.capsules[:]
			self.agent_states = [
				agent_state.copy()
				for agent_state in prev_state.agent_states
			]
			self.killed = prev_state.killed
			self.score = prev_state.score
		else:
			self.layout = None
			self.food = None
			self.capsules = None
			self.agent_states = list()
			self.killed = None
			self.score = 0

		# do anyway
		self._food_eaten = None
		self._food_added = None
		self._capsule_eaten = None
		self._agent_moved = None
		self._lose = False
		self._win = False
		self.score_change = 0

	def __eq__(self, other):
		"""
		Allows two states to be compared.
		"""
		if other is None:
			return False

		if not self.agent_states == other.agent_states:
			return False
		if not self.food == other.food:
			return False
		if not self.capsules == other.capsules:
			return False
		if not self.score == other.score:
			return False
		return True

	def __hash__(self):
		"""
		Allows states to be keys of dictionaries.
		"""

		for i, state in enumerate(self.agent_states):
			try:
				int(hash(state))
			except TypeError as type_error:
				print(type_error)

		return int((hash(tuple(self.agent_states)) +
		            13 * hash(self.food) +
		            113 * hash(tuple(self.capsules))
		            # 7 * hash(self.score))
		            % 1048575))

	def __str__(self):

		# layout properties
		height = self.layout.height
		width = self.layout.width
		wall = self.layout.walls

		# food for this game_state
		food = self.food

		out_map = Matrix(height, width, str, ' ')
		for m in range(height):
			for n in range(width):
				if food[m][n]:
					out_map[m][n] = LayoutCharacters.FOOD
				elif wall[m][n]:
					out_map[m][n] = LayoutCharacters.WALL
				else:
					out_map[m][n] = LayoutCharacters.PATH

		for m, n in self.capsules:
			out_map[m][n] = LayoutCharacters.CAPSULE

		for agent_state in self.agent_states:
			if agent_state is None or agent_state.configuration is None:
				continue

			m, n = agent_state.configuration.position
			# agent_dir = agent_state.configuration.direction

			if agent_state.is_pacman:
				out_map[m][n] = LayoutCharacters.PACMAN
			else:
				out_map[m][n] = LayoutCharacters.GHOST

		return str(out_map) + ("\nScore: %d\n" % self.score)

	def deep_copy(self):
		state = GameState(self)

		state.food = self.food.deep_copy()
		state.layout = self.layout  # .deep_copy()
		state._food_eaten = self._food_eaten
		state._food_added = self._food_added
		state._capsule_eaten = self._capsule_eaten
		state._agent_moved = self._agent_moved
		state._lose = self._lose
		state._win = self._win
		state.score_change = self.score_change

		return state

	def initialize(self, layout, num_ghosts: int):
		"""
		Creates an initial game state from a Layout object.
		"""

		# keep a copy of the layout
		self.layout = layout

		# deep-copy food, capsules here since they will change with the game
		self.food = layout.food.copy()
		self.capsules = layout.capsules[:]

		# start with score = 0
		self.score = 0
		self.score_change = 0

		# maintain state of each agent separately
		self.agent_states = list()
		ghost_count = 0
		for is_pacman, pos in layout.agent_positions:
			if not bool(is_pacman):
				if ghost_count == num_ghosts:
					continue
				else:
					num_ghosts += 1
			self.agent_states.append(AgentState(pos, Direction.STOP, is_pacman))

		# keep record of the agents that have been killed
		self.killed = [False for _ in self.agent_states]

	# #########################################################################

	# static variable keeps track of which states have had get_legal_actions called
	explored = set()

	def get_legal_actions(self, agent_index = 0):
		if self.is_win() or self.is_lose():
			return []

		if agent_index == 0:
			return Pacman.get_legal_actions(self)
		else:
			return Ghost.get_legal_actions(self, agent_index)

	def generate_successor(self, agent_index, action):
		# Check that successors exist
		if self.is_win() or self.is_lose():
			raise Exception('Can\'t generate a successor of a terminal state.')

		# Copy current state
		game_state = GameState(self)

		# Let agent's logic deal with its action's effects on the board
		if agent_index == 0:
			game_state.killed = [False for _ in range(game_state.get_num_agents())]
			Pacman.apply_action(game_state, action)
		else:
			Ghost.apply_action(game_state, action, agent_index)

		# Time passes
		if agent_index == 0:
			game_state.score_change += -TIME_PENALTY  # Penalty for waiting around
		else:
			Ghost.decrement_timer(game_state.agent_states[agent_index])

		# Resolve multi-agent effects
		Ghost.check_death(game_state, agent_index)

		# Book keeping
		game_state._agent_moved = agent_index
		game_state.score += game_state.score_change
		GameState.explored.add(self)
		GameState.explored.add(game_state)
		return game_state

	def get_legal_pacman_actions(self):
		return self.get_legal_actions(0)

	def generate_pacman_successor(self, action):
		return self.generate_successor(0, action)

	def get_pacman_state(self):
		return self.agent_states[0].copy()

	def get_pacman_position(self):
		return self.agent_states[0].get_position()

	def get_ghost_states(self):
		return self.agent_states[1:]

	def get_ghost_state(self, agent_index):
		if agent_index == 0 or agent_index >= self.get_num_agents():
			raise Exception("Invalid index passed to get_ghost_state")
		return self.agent_states[agent_index]

	def get_ghost_position(self, agent_index):
		if agent_index == 0:
			raise Exception("Pacman's index passed to get_ghost_position")
		return self.agent_states[agent_index].get_position()

	def get_ghost_positions(self):
		return [s.get_position() for s in self.get_ghost_states()]

	def get_num_agents(self):
		return len(self.agent_states)

	def get_score(self):
		return float(self.score)

	def get_capsules(self):
		return self.capsules

	def get_num_food(self):
		return self.food.count()

	def get_food(self):
		return self.food

	def get_walls(self):
		return self.layout.walls

	def has_food(self, m, n):
		return self.food[m][n]

	def has_wall(self, m, n):
		return self.layout.walls[m][n]

	def is_lose(self):
		return self._lose

	def is_win(self):
		return self._win


class Game:
	"""
    The Game manages the control flow, soliciting actions from agents.
    """

	def __init__(self, agents, display, rules, starting_index = 0):
		self.agent_crashed = False
		self.agents = agents
		self.display = display
		self.rules = rules
		self.starting_index = starting_index
		self.game_over = False
		self.move_history = []
		self.total_agent_times = [0 for _ in self.agents]
		self.total_agent_time_warnings = [0 for _ in self.agents]
		self.agent_timeout = False
		self.num_moves = 0

	def get_progress(self):
		if self.game_over:
			return 1.0
		else:
			return self.rules.get_progress(self)

	def _agent_crash(self, agent_index):
		"""Helper method for handling agent crashes"""

		traceback.print_exc()
		self.game_over = True
		self.agent_crashed = True

		if agent_index == 0:
			print('Pacman crashed!')
		else:
			print('Ghost crashed!')

	def run(self):
		"""
		Main control loop for game play.
		"""

		self.display.initialize(self.state)
		self.num_moves = 0

		# inform learning agents of the game start
		for i in range(len(self.agents)):
			agent = self.agents[i]

			if not agent:
				# this is a null agent, meaning it failed to load
				# the other team wins
				print("Agent %d failed to load" % i)
				self._agent_crash(i)
				return

			if "register_initial_state" in dir(agent):
				agent.register_initial_state(self.state.deep_copy())

		agent_index = self.starting_index
		num_agents = len(self.agents)

		while not self.game_over:
			# Fetch the next agent
			agent = self.agents[agent_index]

			# Generate an observation of the state
			if 'observation_function' in dir(agent):
				observation = agent.observation_function(self.state.deep_copy())
			else:
				observation = self.state.deep_copy()

			# Solicit an action
			action = agent.get_action(observation)

			# Execute the action
			self.move_history.append((agent_index, action))
			self.state = self.state.generate_successor(agent_index, action)

			# Change the display
			self.display.update(self.state)

			# Allow for game specific conditions (winning, losing, etc.)
			self.rules.process(self.state, self)
			# Track progress
			if agent_index == num_agents + 1:
				self.num_moves += 1
			# Next agent
			agent_index = (agent_index + 1) % num_agents

		# inform a learning agent of the game result
		for agent_index, agent in enumerate(self.agents):
			if "final" in dir(agent):
				try:
					agent.final(self.state)
				except Exception:
					raise

		self.display.finish()
