import numpy

from utility.auxilliary import BooleanMatrix, Direction, Matrix
from utility.datastructures import Edge, Graph, Node


class LayoutCharacters:
	WALL = '*'
	PATH = ' '
	FOOD = '.'
	CAPSULE = 'o'
	PACMAN = 'P'
	GHOST = 'G'

	@classmethod
	def is_wall(cls, char):
		return char == cls.WALL

	@classmethod
	def is_path(cls, char):
		return char == cls.PATH

	@classmethod
	def is_food(cls, char):
		return char == cls.FOOD

	@classmethod
	def is_capsule(cls, char):
		return char == cls.CAPSULE

	@classmethod
	def is_pacman(cls, char):
		return char == cls.PACMAN

	@classmethod
	def is_ghost(cls, char):
		return char == cls.GHOST

	@classmethod
	def as_list(cls):
		return [cls.WALL, cls.PATH, cls.FOOD, cls.CAPSULE, cls.PACMAN, cls.GHOST]


class Layout:
	"""
	Manages the static information about the environment.
	NOTE: the complete geometry of the game is in matrix co-ordinate system NOT in cartesian system.
		  (for ease of implementation) i.e.,
		  - (0,0) is top-left instead of bottom-left.
		  - y-axis (m) increases vertically downwards.
		  - x-axis (n) increases horizontally to the right.

		  *****         --------------------------------
		  * P *         Pacman at (1,2) instead of (2,1)
		  *****         --------------------------------
	"""

	def __init__(self, filename):

		# parse layout file
		self._layout_filename = filename
		self.layout, self.height, self.width = self._parse_layout()

		self.ids = None
		self.food = None
		self.walls = None
		self.capsules = []
		self.agent_positions = []
		self.num_ghosts = 0
		self.total_food = []

		self._process_layout()

	def _parse_layout(self):
		""" Parses the layout file """

		def read_layout_as_matrix(filename):
			""" Return a matrix of characters """

			allowed_chars = LayoutCharacters.as_list()

			# read file
			file = open(filename, 'r', newline = '\n')
			lines = file.readlines()

			char_matrix = list()
			for _, line in enumerate(lines):
				char_array = [ch for ch in line if ch in allowed_chars]
				if len(char_array) > 0:
					char_matrix.append(char_array)
			return char_matrix

		def normalize_layout(matrix):
			""" Make all lines the same length """

			# find max length
			max_length = 0
			for array in matrix:
				length = len(array)
				if max_length < length:
					max_length = length

			# Fill extra space with WALLs
			for idx, _ in enumerate(matrix):
				while len(matrix[idx]) < max_length:
					matrix[idx].append(LayoutCharacters.WALL)

			# return Matrix, M-dim (row), N-dim(column)
			return matrix, len(matrix), max_length

		# parse
		layout = read_layout_as_matrix(self._layout_filename)
		return normalize_layout(layout)

	def _process_layout(self):
		""" Process Layout """

		# generate aux matrices
		self.ids = Matrix(self.height, self.width, int)
		self.food = BooleanMatrix(self.height, self.width)
		self.walls = BooleanMatrix(self.height, self.width)

		# process all characters
		for m in range(self.height):
			for n in range(self.width):
				char = self.layout[m][n]

				self.ids[m][n] = self.width * m + n

				if LayoutCharacters.is_food(char):
					self.food[m][n] = True
				if LayoutCharacters.is_wall(char):
					self.walls[m][n] = True
				if LayoutCharacters.is_capsule(char):
					self.capsules.append((m, n))
				if LayoutCharacters.is_pacman(char):
					self.agent_positions.append((0, (m, n)))
				if LayoutCharacters.is_ghost(char):
					self.agent_positions.append((1, (m, n)))
					self.num_ghosts += 1

		self.agent_positions.sort()
		self.agent_positions = [(i == 0, pos) for i, pos in self.agent_positions]

		# get food locations
		for m in range(self.height):
			for n in range(self.width):
				if self.food[m][n]:
					self.total_food.append((m, n))

		# remove agents from the layout, makes it easy to print
		for m in range(self.height):
			for n in range(self.width):
				char = self.layout[m][n]
				if LayoutCharacters.is_pacman(char) or LayoutCharacters.is_ghost(char):
					self.layout[m][n] = LayoutCharacters.PATH

	def as_graph(self):
		""" Create a graph for the layout """

		graph = Graph()

		# add nodes to the graph
		for m in range(self.height):
			for n in range(self.width):
				node = Node(
					self.ids[m][n],
					mlayout = m,
					nlayout = n
				)
				graph.add_node(node)

		# add edges to the graph
		nodes = graph.get_all_nodes()
		for node in nodes:
			m = node.mlayout
			n = node.nlayout

			# north
			if m - 1 >= 0:
				north_node = graph.get_node_by_nid(self.ids[m - 1][n])
				if north_node is not None:
					edge = Edge(node, north_node, directed = True, action = Direction.NORTH, cost = 1)
					graph.add_edge(edge)

			# east
			if n + 1 < self.width:
				east_node = graph.get_node_by_nid(self.ids[m][n + 1])
				if east_node is not None:
					edge = Edge(node, east_node, directed = True, action = Direction.EAST, cost = 1)
					graph.add_edge(edge)

			# west
			if n - 1 >= 0:
				west_node = graph.get_node_by_nid(self.ids[m][n - 1])
				if west_node is not None:
					edge = Edge(node, west_node, directed = True, action = Direction.WEST, cost = 1)
					graph.add_edge(edge)

			# south
			if m + 1 < self.height:
				south_node = graph.get_node_by_nid(self.ids[m + 1][n])
				if south_node is not None:
					edge = Edge(node, south_node, directed = True, action = Direction.SOUTH, cost = 1)
					graph.add_edge(edge)

		# remove walls nodes from graph
		for node in nodes:
			m = node.mlayout
			n = node.nlayout
			if self.walls[m][n]:
				graph.delete_node(node)

		return graph

	def is_wall(self, position):
		m, n = position
		return self.walls[m][n]

	def get_random_legal_position(self):
		m = numpy.random.choice(self.height)
		n = numpy.random.choice(self.width)
		while self.is_wall((m, n)):
			m = numpy.random.choice(self.height)
			n = numpy.random.choice(self.width)
		return m, n

	def get_num_ghosts(self):
		return self.num_ghosts
