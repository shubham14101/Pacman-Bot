import math

from abstracts.searchproblem import SearchProblem


class PositionalSearchProblem(SearchProblem):
	"""
    This search problem can be used to find paths to a particular point on the pacman board.
    The state space consists of (m,n) positions in a pacman game.
    """

	def __init__(self, game_state, sp = None, gp = None):
		self.search_layout = game_state.layout
		self.search_graph = self.search_layout.as_graph()

		if sp is None and gp is None:
			self.start_pos = game_state.get_pacman_position()
			self.goal_pos = game_state.get_food().as_list()[0]
		else:
			self.start_pos = sp
			self.goal_pos = gp

	def is_goal_state(self, state):
		return self.goal_pos == (state.mlayout, state.nlayout)

	def get_start_state(self):
		return self.search_graph.get_node_by_nid(self.search_layout.ids[self.start_pos[0]][self.start_pos[1]])

	def create_solution(self, path):
		return path, self.cost_of_actions(path)

	def cost_of_actions(self, path):
		return len(path)

	def successor_states(self, state):
		successors = list()

		edges = self.search_graph.get_edges(state)
		for edge in edges:
			nodes = tuple(edge.get_nodes())
			idx = 1 - nodes.index(state)
			neighbor = nodes[idx]
			successors.append((neighbor, edge.action, edge.cost))
		return successors

	def heuristic(self, state):
		pos = (state.mlayout, state.nlayout)

		a = pos[0] - self.goal_pos[0]
		b = pos[1] - self.goal_pos[1]
		return math.sqrt(pow(a, 2) + pow(b, 2))
