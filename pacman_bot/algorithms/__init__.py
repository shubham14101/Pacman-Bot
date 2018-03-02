import enum
from functools import partial

from utility.auxilliary import euclidean_distance, manhattan_distance
from .alphabetapruning import alphabeta as alphabeta_search
from .antcolonyoptimization import AntColonyOptimization
from .astar import search as astarsearch
from .breadthfirst import search as breadthfirstsearch
from .depthfirst import search as depthfirstsearch
from .minimax import minimax as minimax_search
from .uniformcost import search as uniformcostsearch


class Algorithms(enum.Enum):
	BreadthFirstSearch = partial(breadthfirstsearch)
	DepthFirstSearch = partial(depthfirstsearch)
	UniformCostSearch = partial(uniformcostsearch)
	AStarSearch = partial(astarsearch)
	AntColonyOptimization = AntColonyOptimization
	MiniMax = partial(minimax_search)
	AlphaBeta = partial(alphabeta_search)


class SearchHeuristics(enum.Enum):
	NullHeuristic = partial(lambda sp, n: 0.0)
	EuclideanHeuristic = partial(lambda sp, n: euclidean_distance((n.mlayout, n.nlayout), sp.goal_pos))
	ManhattanHeuristic = partial(lambda sp, n: manhattan_distance((n.mlayout, n.nlayout), sp.goal_pos))
