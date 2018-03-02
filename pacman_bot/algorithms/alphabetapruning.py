def alphabeta(game_state, agent_idx, evaluation_function, alpha, beta, current_depth, max_depth):
	"""
	Controlling function for MiniMax Algorithm.
	This function decides controls whether to call the min_value or the max_value function
	for a particular agent. Also, decides when to increase the depth in MiniMax tree.

	:param game_state: current state of the game. [@type pacman.game.GameState]
	:param agent_idx: current agent index in agents list of game_state
	:param evaluation_function: _self-descriptive
	:param alpha: best max-value yet
	:param beta: best min-value yet
	:param current_depth: current depth in the MiniMax tree
	:param max_depth: max depth to go in the MiniMax tree
	:return: evaluated value for a state, action to take to achieve it
	"""

	# if this is the last agent in the list,
	#   - increase depth
	#   - change index back to 0 (pacman)
	if agent_idx >= game_state.get_num_agents():
		current_depth = current_depth + 1
		agent_idx = 0

	# if max-depth reached, evaluate current state
	# i.e., start calculating bottom-up in the MiniMax tree
	if current_depth >= max_depth:
		value = evaluation_function(game_state)
		return value, None

	# get a list of all possible actions
	legal_actions = game_state.get_legal_actions(agent_idx)

	# if no further action is possible, evaluate current state
	# i.e., start calculating bottom-up in the MiniMax tree
	if len(legal_actions) == 0:
		value = evaluation_function(game_state)
		return value, None

	# maximize for pacman (idx = 0), minimize for ghosts (idx > 0)
	if agent_idx == 0:
		return max_value(game_state, agent_idx, evaluation_function, alpha, beta, current_depth, max_depth)
	else:
		return min_value(game_state, agent_idx, evaluation_function, alpha, beta, current_depth, max_depth)


def max_value(game_state, agent_idx, evaluation_function, alpha, beta, current_depth, max_depth):
	"""
	Chooses an action that maximize the evaluation function's value.
	"""

	# start with the least possible value
	node_value = (-float("inf"), None)

	# get a list of all possible actions
	possible_actions = game_state.get_legal_actions(agent_idx)

	# for each possible action
	#   - generate next state
	#   - continue down the MiniMax tree, and get the best possible evaluation
	#       - at each step check against beta
	#       - if current-max > beta, prune
	#       - else update alpha
	#   - choose the action that leads to the maximum of these values
	for action in possible_actions:
		successor_state = game_state.generate_successor(agent_idx, action)
		evaluation = alphabeta(successor_state, agent_idx + 1, evaluation_function, alpha, beta, current_depth,
		                       max_depth)
		value = evaluation[0]

		# select max value as node value, with corresponding action
		if value >= node_value[0]:
			node_value = (value, action)

		current_max = node_value[0]
		if current_max > beta:
			return node_value

		alpha = max(current_max, beta)

	return node_value


def min_value(game_state, agent_idx, evaluation_function, alpha, beta, current_depth, max_depth):
	"""
	Chooses an action that minimizes the evaluation function's value.
	"""

	# start with the largest possible value
	node_value = (float("inf"), None)

	# get a list of all possible actions
	possible_actions = game_state.get_legal_actions(agent_idx)

	# for each possible action
	#   - generate next state
	#   - continue down the MiniMax tree, and get the best possible evaluation
	#       - at each step check against beta
	#       - if current-max > beta, prune
	#       - else update alpha
	#   - choose the action that leads to the maximum of these values
	for action in possible_actions:
		successor_state = game_state.generate_successor(agent_idx, action)
		evaluation = alphabeta(successor_state, agent_idx + 1, evaluation_function, alpha, beta, current_depth,
		                       max_depth)
		value = evaluation[0]

		# select min value as node value, with corresponding action
		if value <= node_value[0]:
			node_value = (value, action)

		current_min = node_value[0]
		if current_min < alpha:
			return node_value

		beta = min(current_min, alpha)

	return node_value
