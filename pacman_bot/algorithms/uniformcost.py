from abstracts.searchproblem import SearchProblem
from utility.datastructures import PriorityQueue
from utility.auxilliary import traverse_path


def search(searchproblem: SearchProblem, *args, **kwargs) -> [list, None]:
	frontier = PriorityQueue(entry_generator = lambda x: (x[0], x))
	explored = set()
	meta_map = dict()
	node_distance_track = dict()

	startnode = searchproblem.get_start_state()
	meta_map[startnode] = (None, None)
	frontier.push((0, startnode))
	node_distance_track[startnode] = 0

	if searchproblem.is_goal_state(startnode):
		return list()

	while not frontier.is_empty():

		priority, node = frontier.pop()

		if priority > node_distance_track[node]:
			continue
		if searchproblem.is_goal_state(node):
			action_list = traverse_path(node, meta_map)
			return action_list

		explored.add(node)

		for (child, action, cost) in searchproblem.successor_states(node):
			if child in explored:
				continue

			if child not in node_distance_track or node_distance_track[child] > priority + cost:

				frontier.push((priority + cost, child))
				node_distance_track[child] = priority + cost
				meta_map[child] = (node, action)
	return None
