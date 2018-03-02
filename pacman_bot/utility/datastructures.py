import heapq

from .error import raise_unexpected_behaviour


class Stack:
	"""
	LIFO container.
	"""

	def __init__(self):
		self._container = list()

	def push(self, item):
		"""
		:param item: object to push to end of stack.
		"""
		self._container.append(item)

	def pop(self):
		"""
		:return: the most recently pushed item from the stack.
		"""
		return self._container.pop()

	def is_empty(self):
		"""
		:return: `True` if stack is empty.
		"""
		return len(self._container) == 0

	def __str__(self):
		return str(self._container)

	def __contains__(self, item):
		return item in self._container


class Queue:
	"""
	FIFO container.
	"""

	def __init__(self):
		self._container = list()

	def push(self, item):
		"""
		:param item: object to push to beginning of queue.
		"""
		self._container.insert(0, item)

	def pop(self):
		"""
		:return: the least recently pushed item from the queue.
		"""
		return self._container.pop()

	def is_empty(self):
		"""
		:return: `True` if queue is empty.
		"""
		return len(self._container) == 0

	def __str__(self):
		return str(self._container)

	def __contains__(self, item):
		return item in self._container


class PriorityQueue:
	"""
	Implements a PriorityQueue based on the Entry Generator provided.
	By default a min-heap.
	"""

	def __init__(self, entry_generator = lambda x: (x, x)):
		"""
		:param entry_generator: a function that takes in item and returns (priority, item)
		"""
		self._container = list()
		self._count = 0
		self._entry_generator = entry_generator

	def push(self, item):
		entry = self._entry_generator(item)
		heapq.heappush(self._container, entry)
		self._count += 1

	def pop(self):
		(_, item) = heapq.heappop(self._container)
		return item

	def is_empty(self):
		return len(self._container) == 0

	def get_heap(self):
		return [item for (_, item) in self._container]

	def __str__(self):
		repr_list = self.get_heap()
		return str(repr_list)

	def __contains__(self, item):
		return item in self.get_heap()


class Node:
	"""
	a vertice of the graph.
	"""

	def __init__(self, nid, **kwargs):
		"""
		create a new node.
		:param nid: unique node identifier
		:param kwargs: extra properties to set for a node
		"""

		if nid is None:
			raise_unexpected_behaviour()

		self._nid = nid
		for key in kwargs:
			setattr(self, key, kwargs[key])

	def __str__(self):
		return str(self._nid)

	def __hash__(self):
		return hash(str(self))

	def __eq__(self, other):
		if not isinstance(other, Node):
			return False

		if self._nid == other._nid:
			return True
		return False

	def __cmp__(self, other):
		return self.get_node_id() - other.get_node_id()

	def __lt__(self, other):
		return self.get_node_id() < other.get_node_id()

	def __le__(self, other):
		return self.get_node_id() <= other.get_node_id()

	def __gt__(self, other):
		return self.get_node_id() > other.get_node_id()

	def __ge__(self, other):
		return self.get_node_id() >= other.get_node_id()

	def get_node_id(self):
		return self._nid


class Edge:
	"""
	an edge in the graph.
	"""

	def __init__(self, node1, node2, directed: bool = False, **kwargs):
		"""
		create an edge.
		"""

		self._directed = directed
		self._edge = (node1, node2)
		for key in kwargs:
			setattr(self, key, kwargs[key])

	def __str__(self):
		return str(self._edge[0]) + " --> " + str(self._edge[1])

	def __eq__(self, other):
		if not isinstance(other, Edge):
			return False

		if self._directed:
			if len(self._edge) == len(other._edge):
				for idx, node in enumerate(self._edge):
					if other._edge[idx] != node:
						return False
				return True
		else:
			if len(self._edge) == len(other._edge):
				for _, node in enumerate(self._edge):
					if node not in other._edge:
						return False
				return True

		return False

	def __contains__(self, item: Node):
		if not isinstance(item, Node):
			return False
		return item in self._edge

	def __hash__(self):
		lst = list(self._edge)
		lst.sort()
		lst = tuple(lst)
		return hash(lst)

	def get_nodes(self):
		return self._edge

	def is_directed(self):
		return self._directed

	# #########################################################################


class Graph:
	"""
	implements a graph data structure.
	"""

	def __init__(self):
		self._container = dict()

	def add_node(self, node):
		"""
		Adds a Graph.Node to the graph
		:param node: node object
		"""

		if (node is None) or (not isinstance(node, Node)):
			return

		if node not in self._container:
			self._container[node] = list()

	def add_edge(self, edge: Edge):
		"""
		Adds an Graph.Edge to the graph.
		:param edge: edge object
		"""

		if (edge is None) or (not isinstance(edge, Edge)):
			return

		nodes = edge.get_nodes()
		if len(nodes) != 2:
			return

		for node in nodes:
			if node not in self._container:
				return

		if edge.is_directed():
			self._container[nodes[0]].append(edge)
		else:
			self._container[nodes[0]].append(edge)
			self._container[nodes[1]].append(edge)

	def delete_edge(self, edge: Edge):
		"""
		Removes an edge from graph
		:param edge:
		"""

		if (edge is None) or (not isinstance(edge, Edge)):
			return

		nodes = edge.get_nodes()
		if len(nodes) != 2:
			return

		if edge.is_directed():
			if nodes[0] in self._container:
				self._container[nodes[0]].remove(edge)
		else:
			if nodes[0] in self._container:
				self._container[nodes[0]].remove(edge)
			if nodes[1] in self._container:
				self._container[nodes[1]].remove(edge)

	def delete_node(self, node):

		if (node is None) or (not isinstance(node, Node)):
			return

		if node in self._container:
			for edge in self._container[node]:
				self.delete_edge(edge)

		self._container.pop(node, None)

		for other in self.get_all_nodes():
			for edge in self._container[other]:
				if node in edge:
					self.delete_edge(edge)

	def get_all_nodes(self):
		return list(self._container.keys())

	def get_all_edges(self):
		edges = set()
		for node in self.get_all_nodes():
			for edge in self._container[node]:
				edges.add(edge)
		return list(edges)

	def get_node_by_nid(self, nid):
		nodes = self.get_all_nodes()
		for node in nodes:
			if node.get_node_id() == nid:
				return node
		return None

	def get_edges(self, node):
		if node in self._container:
			return self._container[node]

	def get_edge(self, node1, node2):
		if node1 in self._container:
			edges = self._container[node1]
			for edge in edges:
				if node2 in edge:
					return edge
		return None
