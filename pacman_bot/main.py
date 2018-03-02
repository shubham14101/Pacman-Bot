import argparse
import enum
import os

from pacman import agents
from pacman.game import ClassicGameRules
from pacman.graphics import NullGraphics, TextGraphics
from pacman.layout import Layout

# #############################################################################
# Runs a Pacman game.
# #############################################################################

# ### Some constants

LAYOUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pacman/layout_files')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pacman/models')
DEFAULT_LAYOUT = 'positional_search.lay'


# ### Enums
class PacmanAgents(enum.Enum):
	PositionalSearchAgent = agents.PositionalSearchAgent
	FoodSearchAgent = agents.FoodSearchAgent
	ReflexAgent = agents.ReflexAgent
	MiniMaxAgent = agents.MinimaxAgent
	AlphaBetaAgent = agents.AlphaBetaAgent
	ExpectiMaxAgent = agents.ExpectimaxAgent
	QLearningAgent = agents.QLearningAgent


class GhostAgents(enum.Enum):
	RandomAgent = agents.RandomAgent
	ChasingAgent = agents.ChasingGhostAgent


def parse_commands():
	"""
	Parses the command line arguments and returns a arguments object.
	"""

	parser = argparse.ArgumentParser(
		description = "Pacman-Bot, a pacman simulator that implements various AI techniques" "\n"
		              "-- by Gursimran Singh (2014041), Shubham Maheshwari(2014101)"
	)

	parser.add_argument(
		'--num-games', '-N',
		dest = 'num_games',
		action = 'store',
		type = int,
		default = 1,
		help = "number of GAMES to run."
	)
	parser.add_argument(
		'--num-training', '-n',
		dest = 'num_training',
		action = 'store',
		type = int,
		default = 0,
		help = "number of GAMES to train. used for the purpose of training!"
	)
	parser.add_argument(
		'--layout', '-l',
		dest = 'layout',
		action = 'store',
		type = str,
		default = DEFAULT_LAYOUT,
		help = "LAYOUT_FILEPATH relative to `pacman_bot/pacman/layout_files/`"
	)
	parser.add_argument(
		'--pacman', '-p',
		dest = 'pacman',
		action = 'store',
		type = str,
		choices = [agent_type.name for agent_type in PacmanAgents],
		default = PacmanAgents.PositionalSearchAgent.name,
		help = "PACMAN agent type"
	)
	parser.add_argument(
		'--pacman-args', '-pargs',
		dest = 'pacman_args',
		action = 'store',
		type = str,
		help = "Comma separated values of arguments for a pacman agent.py (no spaces)"
	)
	parser.add_argument(
		'--ghost', '-g',
		dest = 'ghost',
		action = 'store',
		type = str,
		choices = [agent_type.name for agent_type in GhostAgents],
		default = GhostAgents.RandomAgent.name,
		help = "GHOST agent type"
	)
	parser.add_argument(
		'--num-ghosts', '-k',
		dest = 'num_ghosts',
		action = 'store',
		type = int,
		default = 1,
		help = "maximum number of GHOST agents"
	)
	parser.add_argument(
		'--frames-per-second', '-fps',
		dest = 'fps',
		action = 'store',
		type = int,
		default = 1,
		help = "FRAMES_PER_SECOND (display option)"
	)

	args = parser.parse_args()
	return args


def process_commands(rargs):
	"""
	Processes the parsed arguments for sanitation.
	"""

	def parse_agent_args(args):
		if args is None:
			return dict()

		args = args.split(',')
		opts = dict()
		for arg in args:
			if '=' in arg:
				key, val = arg.split('=')
			else:
				key, val = arg, 1
			opts[key] = val
		return opts

	# #################################

	args = dict()

	# process layout file
	rargs.layout = os.path.join(LAYOUT_DIR, rargs.layout)
	if not os.path.exists(rargs.layout) or not os.path.isfile(rargs.layout):
		raise FileNotFoundError('Provided `layout` file not found!')

	args['layout'] = Layout(rargs.layout)

	# process pacman agent
	rargs.pacman = PacmanAgents[rargs.pacman]
	pac_args = parse_agent_args(rargs.pacman_args)

	pacman_type = rargs.pacman.value

	if rargs.pacman == PacmanAgents.QLearningAgent:
		pac_args['num_training'] = rargs.num_training

		pac_args.setdefault('save', None)
		pac_args.setdefault('load', None)

		save = pac_args['save']
		if save is not None:
			save = os.path.join(MODELS_DIR, save)

		load = pac_args['load']
		if load is not None:
			load = os.path.join(MODELS_DIR, load)

		pac_args['save'] = save
		pac_args['load'] = load

		print(pac_args['save'], pac_args['load'])

	args['pacman'] = pacman_type(**pac_args)

	# process ghost agents
	rargs.ghost = GhostAgents[rargs.ghost]
	ghost_type = rargs.ghost.value
	args['ghosts'] = [ghost_type(i + 1) for i in range(rargs.num_ghosts)]

	# process number of games
	args['num_games'] = rargs.num_games
	args['num_training'] = rargs.num_training

	# process display options
	args['display'] = TextGraphics(speed = rargs.fps)

	return args


def run_pacman(layout: Layout, display, pacman, ghosts, num_games, num_training):
	rules = ClassicGameRules()
	games = []

	null_display = NullGraphics()

	for i in range(num_games):
		be_quiet = i < num_training
		if be_quiet:
			game_display = null_display
			rules.quiet = True
		else:
			game_display = display
			rules.quiet = False

		game = rules.new_game(layout, pacman, ghosts, game_display, be_quiet)
		game.run()

		if not be_quiet:
			games.append(game)

		# if record:
		# 	import time, cPickle
		# 	fname = ('recorded-game-%d' % (i + 1)) + '-'.join([str(t) for t in time.localtime()[1:6]])
		# 	f = file(fname, 'w')
		# 	components = {'layout': layout, 'actions': game.moveHistory}
		# 	cPickle.dump(components, f)
		# 	f.close()

	if (num_games - num_training) > 0:
		scores = [game.state.get_score() for game in games]
		wins = [game.state.is_win() for game in games]
		win_rate = wins.count(True) / float(len(wins))
		print('Average Score:', sum(scores) / float(len(scores)))
		print('Scores:       ', ', '.join([str(score) for score in scores]))
		print('Win Rate:      %d/%d (%.2f)' % (wins.count(True), len(wins), win_rate))
		print('Record:       ', ', '.join([['Loss', 'Win'][int(w)] for w in wins]))

	return games


if __name__ == '__main__':
	raw_args = parse_commands()
	game_args = process_commands(raw_args)
	run_pacman(**game_args)
