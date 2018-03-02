from abstracts.searchproblem import SearchProblem
from utility.auxilliary import traverse_path
from utility.datastructures import PriorityQueue


def search(searchproblem: SearchProblem, heuristic = lambda x, y: 0.0, *args, **kwargs) -> [list, None]:
	"""
	A Star Search - lowest cost till current plus lowest heuristic first.
	Made in accordance to general graph search pseudo-code in textbook - Fig 3.7

	# f-value = g-value + h-value
	# f-value = priority
	# g-value = path cost
	# h-value = heuristic

	:param searchproblem: instance of a search problem
	:param heuristic: a heuristic function to use -> h(searchproblem, node)
	:return: a list of actions to reach from start to goal node if found, else None
	"""

	# min-heap frontier, takes in entry as (f-value, node)
	frontier = PriorityQueue(entry_generator = lambda x: x)
	# visited set
	explored = set()
	# to keep track of each node's meta information: (parent, action, g-value, f-value)
	meta_map = dict()

	# function to calculate f-value
	def f(g, h):
		return g + h

	# default meta value
	__default_meta = (None, None, float("inf"), float("inf"))

	# start node
	startnode = searchproblem.get_start_state()
	gvalue = 0.0
	hvalue = heuristic(searchproblem, startnode)
	fvalue = f(gvalue, hvalue)
	meta_map[startnode] = (None, None, fvalue, gvalue)
	frontier.push((fvalue, startnode))

	while not frontier.is_empty():

		node = frontier.pop()
		path_cost = meta_map[node][3]

		if searchproblem.is_goal_state(node):
			action_list = traverse_path(node, meta_map)
			return action_list

		for (child, action, cost) in searchproblem.successor_states(node):

			if child in explored:
				continue

			gvalue = path_cost + cost
			hvalue = heuristic(searchproblem, child)
			fvalue = f(gvalue, hvalue)

			if child not in frontier:
				frontier.push((fvalue, child))

			# set a default value for the child, if does not already exists
			meta_map.setdefault(child, __default_meta)

			# if current f-value is less than the old f-value, update
			if fvalue < meta_map[child][2]:
				meta_map[child] = (node, action, fvalue, gvalue)

		explored.add(node)

	return None
