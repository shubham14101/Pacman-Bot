import inspect
import sys


def raise_unexpected_behaviour():
	file_name = inspect.stack()[1][1]
	line = inspect.stack()[1][2]
	method = inspect.stack()[1][3]

	print("*** Unexpected Behaviour Found: %s at line %s of %s" % (method, line, file_name))
	sys.exit(1)
