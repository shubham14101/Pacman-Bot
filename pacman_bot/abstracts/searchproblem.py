from abc import ABC, abstractmethod


class SearchProblem(ABC):
	"""
	This abstract class structures a Generic Search Problem.
	The methods are defined in accordance to textbook: Artificial Intelligence, A Modern Approach
	"""

	@abstractmethod
	def get_start_state(self):
		"""
		:return: initial state of the search algorithm
		"""
		raise NotImplementedError()

	@abstractmethod
	def is_goal_state(self, state):
		"""
		:param state: search state
		:return: true iff goal else false
		"""
		raise NotImplementedError()

	@abstractmethod
	def successor_states(self, state):
		"""
		:param state: search state
		:return: a list of possible successor (state, action, cost)
		"""
		raise NotImplementedError()

	@abstractmethod
	def create_solution(self, path):
		"""
		:param path: an ordered list of states to follow to reach the goal state.
		:return: the required form of solution
		"""
		raise NotImplementedError()
