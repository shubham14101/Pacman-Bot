import math
from timeit import default_timer

import numpy

from utility.datastructures import Edge, Graph, Node


class Ant:
	def __init__(self, start_location, env_graph: Graph, alpha, beta, first_pass = False):
		self.start_location = start_location
		self.allowed_locations = env_graph.get_all_nodes()
		self.route_taken = list()
		self.distance_travelled = 0.0
		self.env_graph = env_graph
		self.alpha = alpha
		self.beta = beta
		self.first_pass = first_pass
		self.tour_completed = False

		# append the starting location to route
		self._update_route_taken(start_location)
		self.current_location = start_location

	def _choose_next_node(self):
		"""
		Chooses the next node based on the pheromone levels on available paths.
		:return:
		"""

		# We have no information about pheromone distribution as of now
		# choose a path randomly with equal probability
		if self.first_pass:
			return numpy.random.choice(self.allowed_locations)

		# traversal probability is calculated according to
		#   p(i,j) = (T(i,j)^alpha * n(i,j)^beta) / ∑[k = allowed nodes](T(i,k)^alpha * n(i,k)^beta)
		#   - T = Tau, n = eta
		#   - NOTE: n = 1 / distance

		# probability of choosing a path to a node
		traversal_probability = list()

		for idx, next_node in enumerate(self.allowed_locations):
			edge = self.env_graph.get_edge(self.current_location, next_node)
			pheromone_level = edge.pheromone_level
			weight = edge.weight

			# calculate all numerators
			p = math.pow(pheromone_level, self.alpha) * math.pow(1 / weight, self.beta)
			traversal_probability.append(p)

		# total pheromone level on all the paths
		# used to compute the traversal probabilities
		total_pheromone_level = sum(traversal_probability)

		# change numerators to actual probabilities
		for idx in range(len(traversal_probability)):
			traversal_probability[idx] = traversal_probability[idx] / total_pheromone_level

		# choose from allowed next locations according to this probability
		next_node = numpy.random.choice(self.allowed_locations, p = traversal_probability)
		return next_node

	def _move_to_node(self, current_node, next_node):
		"""
		Move to next node from current node.
		:param current_node:
		:param next_node:
		:return:
		"""

		self._update_route_taken(next_node)
		self._update_distance_travelled(current_node, next_node)
		self.current_location = next_node

	def _update_route_taken(self, location):
		"""
		Add new location to route taken.
		Remove new location from the list of possible next locations.
		:param location:
		:return:
		"""


		self.route_taken.append(location)
		self.allowed_locations.remove(location)

	def _update_distance_travelled(self, current_node, next_node):
		"""
		Update the distance with the weight of the edge traversed
		:param current_node:
		:param next_node:
		:return:
		"""

		weight = self.env_graph.get_edge(current_node, next_node).weight
		self.distance_travelled += float(weight)

	def run_ant_tour(self):
		"""
		Moves an ant for a complete sub-iteration (until it completes the route)
		:return:
		"""

		while self.allowed_locations:
			next_node = self._choose_next_node()
			self._move_to_node(self.current_location, next_node)

		# finally go back to the initial node (to complete circuit)
		# self.allowed_locations.append(self.route_taken[0])
		# next_node = self.route_taken[0]
		# self._move_to_node(self.current_location, next_node)
		self.tour_completed = True

	def get_route(self):
		"""
		If the tour is completed, return the route taken
		:return:
		"""

		if self.tour_completed:
			return self.route_taken
		return None

	def get_distance(self):
		"""
		If the tour is complete, return the distance travelled
		:return:
		"""

		if self.tour_completed:
			return self.distance_travelled
		return None


class AntColonyOptimization:
	def __init__(self, model, start = None, ant_count = 100, alpha = 1.0, beta = 1.0, pheromone_evaporation_coeff = 0.3,
	             pheromone_deposition_constant = 500, max_iterations = -1, timeout_sec = 10):

		self.model = model
		self.env_graph = self._create_graph(model)

		self.start = start
		self.ant_count = int(ant_count)

		self.alpha = float(alpha)
		if self.alpha < 0:
			raise ValueError("alpha > 0")

		self.beta = float(beta)
		if self.beta < 1:
			raise ValueError("beta >= 1")

		self.pheromone_evaporation_coeff = float(pheromone_evaporation_coeff)
		self.pheromone_deposition_constant = float(pheromone_deposition_constant)
		self.max_iterations = int(max_iterations)
		self.timeout_sec = float(timeout_sec)

		self.first_pass = True
		self.ants = self._create_ants()
		self.best_tour = None

	def _create_graph(self, model: dict) -> Graph:
		"""
		Creates a env_graph from the tsp model.
		:param model:
		:return:
		"""

		graph = Graph()

		keys = list(model.keys())

		for idx, pos in enumerate(keys):
			node = Node(str(pos), name = str(pos), mlayout = pos[0], nlayout = pos[1])
			graph.add_node(node)

		for idx1, pos1 in enumerate(keys):
			node1 = graph.get_node_by_nid(str(pos1))
			for idx2, tup in enumerate(model[pos1]):
				pos2, _, cost = tup
				node2 = graph.get_node_by_nid(str(pos2))
				edge = Edge(node1, node2, directed = False, weight = cost, pheromone_level = 0.0)
				graph.add_edge(edge)

		return graph

	def _create_ants(self):
		"""
		Create ants for propagation.
		:return:
		"""

		ants = list()
		node = self.env_graph.get_node_by_nid(str(self.start))

		for idx in range(self.ant_count):
			ants.append(Ant(node, self.env_graph, self.alpha, self.beta, self.first_pass))

		return ants

	def _update_pheromone_values(self, ant):
		"""
		Updates the values of pheromones on each edge of the graph.
		:param ant:
		:return:
		"""

		route = ant.get_route()
		for i in range(len(route) - 1):
			edge = self.env_graph.get_edge(route[i], route[i + 1])
			updated_pheromone_level = self.pheromone_deposition_constant / ant.get_distance()
			edge.pheromone_level += updated_pheromone_level

	def _decay_pheromones(self):
		"""
		Refers to the timely decay of pheromones on each edge.
		:return:
		"""

		for edge in self.env_graph.get_all_edges():
			current_pheromone_level = edge.pheromone_level

			# decay according to (1 - rho) * tau
			edge.pheromone_level = (1 - self.pheromone_evaporation_coeff) * current_pheromone_level

	def run_ant_colony_optimization(self):
		"""
		Run the main ant colony optimization algorithm.
		:return:
		"""

		# looping conditions
		start_time = default_timer()
		iteration_counter = 0

		loop_continue_condition = True
		loop_continue_condition = loop_continue_condition and (((
			                                                        default_timer() - start_time) < self.timeout_sec) if self.timeout_sec >= 0 else True)
		loop_continue_condition = loop_continue_condition and ((
			                                                       iteration_counter < self.max_iterations) if self.max_iterations >= 0 else True)

		while loop_continue_condition:

			iteration_counter += 1

			# decay the pheromone values
			self._decay_pheromones()

			# run the tour for each ant
			for ant in self.ants:
				ant.run_ant_tour()

				# update the pheromones value as defined by the ant tour
				self._update_pheromone_values(ant)

				# store the best path
				if not self.best_tour or self.best_tour[1] > ant.get_distance():
					self.best_tour = (ant.get_route(), ant.get_distance())

			if self.first_pass:
				self.first_pass = False

			self.ants = self._create_ants()

			# re-calculate looping conditions
			loop_continue_condition = True
			loop_continue_condition = loop_continue_condition and (((
				                                                        default_timer() - start_time) < self.timeout_sec) if self.timeout_sec >= 0 else True)
			loop_continue_condition = loop_continue_condition and ((
				                                                       iteration_counter < self.max_iterations) if self.max_iterations >= 0 else True)

		return self.best_tour
