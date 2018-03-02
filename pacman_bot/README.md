# **CSE643 - Artificial Intelligence**
## **Introduction**
**Pacman-Bot** is a simple project that simulates the well known game of Pacman and the Ghosts. Along the way, we try to implement various AI techniques such as *searching*, *multi-agent search*, and *reinforcement learning*.  
**Authors**:  
1\. Gursimran Singh (2014041)  
2\. Shubham Maheshwari (2014101)


## **Detail Description**
**Pacman-Bot** defines the game of Pacman and the Ghosts using various abstractions.

### Layouts
*Layout* refers to the Pacman World or more precisely the Pacman game board. The Pacman board for the game is defined in a `.lay` file (just a simple text file with an intutive extension). All the layouts for the project are defined in `pacman_bot/pacman/layouts/` directory.  
The name of the layout file to use during the game can be provided using command line arguments `--layout` or `-l`. The name/path provided must be relative to the layouts directory.  
A custom layout can be easily defined. A layout can contain only the following characters:  
`*` (Wall), `<space>` (Empty and Open Path), `.` (Food), `P` (Pacman), `G` (Ghost).

### Agent
An agent refers to either 'Pacman' or a 'Ghost'. The agent is intelligent entity and can exhibit different kinds of behaviours hence there are multiple types of agents that one can use in the game. All agent types try to solve different types of problems.  
To change 'Pacman' agent use `-p <PacmanAgents>` or `--pacman <PacmanAgent>` in arguments.
To change 'Ghost' agents use `-g <PacmanAgents>` or `--ghost <GhostAgent>` in arguments.  
Agent arguments can be supplied using `-pargs`.

#### Supported Pacman Agent Types
- **Positional Search Agent**  
	Searches for a path (hopefully shortest, depends on the search function) from Pacman's location to a single food particle. Does not lookout for Ghosts. Do not use on layouts with more than one food particle. Do not use with Ghosts.  
	Use `-p PositionalSearchAgent`.  
	Arguments
	- search_function_name:
		- `BreadthFirstSearch`
		- `DepthFirstSearch`
		- `UniformCostSearch`
		- `AStarSearch`.
	- heuristic_name (for AStarSearch)
		- `NullHeuristic`
		- `EuclideanHeuristic`
		- `ManhattanHeuristic`  


- **Food Search Agent**  
	Searches for the shortest path that will collect all the food particles starting from Pacman's current location. Uses AStarSearch. Does not lookout for Ghosts. Do not use with Ghosts. [Uses ACO to approximate TSP path, hence takes time (~10sec)]  
	Use `-p FoodSearchAgent`.
	Arguments
	- search_function_name:
		- `BreadthFirstSearch`
		- `DepthFirstSearch`
		- `UniformCostSearch`
		- `AStarSearch`.
	- heuristic_name (for AStarSearch)
		- `NullHeuristic`
		- `EuclideanHeuristic`
		- `ManhattanHeuristic`
	- ant_count
	- alpha
	- beta
	- pheromone_evaporation_coeff
	- pheromone_deposition_constant
	- max_iterations
	- timeout_sec


- **Reflex Agent**  
	Chooses its next action randomly, slightly influenced by an evaluation function (reflex function).  
	Use `-p ReflexAgent`.


- **MiniMax Agent**  
	Chooses the best possible action for Pacman on the assumption that the Ghosts are playing optimally.  
	Use `-p MiniMaxAgent`.  
	Arguments
	- evaluation_function_name
		- `ScoreEvaluationFunc`
		- `FoodCountEvaluationFunc`
	- depth


- **AlphaBeta Pruning Agent**  
	Improves MiniMax runtime by pruning subtrees that are known to not provide a solution.  
	Use `-p AlphaBetaAgent`.  
	Arguments
	- evaluation_function_name
		- `ScoreEvaluationFunc`
		- `FoodCountEvaluationFunc`
	- depth  


- **ExpectiMax Agent**  
	Does not assume that Ghosts are playing optimally. Uses probabilities.  
	Use `-p ExpectiMaxAgent`.  
	Arguments
	- evaluation_function_name:
		- `ScoreEvaluationFunc`
		- `FoodCountEvaluationFunc`
	- depth


#### Supported Ghost Agent Types

- **Random Agent**  
	Chooses next action randomly.  
	Use `-g RandomAgent`.
- **Chasing Agent**  
	Always tries to chase Pacman with the highest probability, if not scared (Pacman hasn't eaten a capsule). If scared, tries to run away from Pacman.
	Use `-g ChasingAgent`.


## **How to run**
This project was developed using `python3`. Please create a virtual environment and install all the requirements before running.

### Setup Virtual Environment
1. Change Directory to Project folder -  
`cd path/to/pacman_bot`
2. Create a Virtual Environment -  
`python3 -m venv .`
3. Activate Virtual Environment -  
`source bin/activate`
4. Install Requirements -  
`pip install -r requirements.txt`

### Run Pacman
1. `python3 main.py <arguments>`

### Some Example commands
- `python3 main.py -l positional_search.lay -pargs search_function_name=AStarSearch,heuristic_name=EuclideanHeuristic`
- `python3 main.py -l multi_agent.lay -p AlphaBetaAgent -pargs evaluation_function_name=FoodCountEvaluationFunc,depth=5`
