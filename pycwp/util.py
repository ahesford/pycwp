'''
General-purpose, non-numerical routines.
'''
import math
from itertools import islice

from . import cutil

class bidict(dict):
	'''
	Extends a dictionary to simultaneously keep an inverse dictionary for
	two-way lookups.
	'''
	def __init__(self, *args, **kwargs):
		super(bidict, self).__init__(*args, **kwargs)
		self.inverse = {}
		for key, value in self.iteritems():
			self.inverse.setdefault(value, []).append(key)

	def __setitem__(self, key, value):
		super(bidict, self).__setitem__(key, value)
		self.inverse.setdefault(value, []).append(key)

	def __delitem__(self, key):
		self.inverse.setdefault(self[key],[]).remove(key)
		if self[key] in self.inverse and not self.inverse[self[key]]:
			del self.inverse[self[key]]
		super(bidict, self).__delitem__(key)


class lazy_property(object):
	'''
	A decorator to enable lazy evaluation of a read-only property.
	Property should be immutable as it is replaced by the computed value.
	'''
	def __init__(self, fget):
		self.fget = fget
		self.func_name = fget.__name__

	def __get__(self, obj, cls):
		if obj is None: return None
		value = self.fget(obj)
		setattr(obj, self.func_name, value)
		return value


def zeropad(d, m):
	'''
	Given a nonnegative integer d less than an integer m, return a string
	representation s of d with enough leading zeros prepended such that all
	integers not more than m can be displayed in strings of length(s).

	Examples:
	  zeropad(3, 999) -> '003'
	  zeropad(999, 999) -> '999'
	  zeropad(12, 1000) -> '0012'
	'''
	digs = max(d, m)
	return '{0:0{digits}d}'.format(d, digits=cutil.numdigits(digs))


def grouplist(lst, n):
	'''
	Create a generator that returns a sequence of n-tuples taken from
	successive groups of n values from the iterable lst.

	The final tuple may have a dimensionality less than n if the length of
	the iterable is not evenly divided by n.
	'''
	iterable = iter(lst)
	while True:
		yield tuple(islice(iterable, n)) or iterable.next()


def printflush(string):
	'''
	Print a string, without a newline, and flush stdout.
	'''
	from sys import stdout
	print string,
	stdout.flush()


class ProgressBar:
	'''
	Generate a string representing a progress bar on the console.
	'''
	def __init__(self, bounds=[0, 100], width=30):
		'''
		Create a progress bar of width characters to show the position
		of a counter within the specified bounds.
		'''
		# Copy the bounds and bar width
		self.bounds = bounds[:]
		self.range = bounds[1] - bounds[0]
		self.barwidth = width

		# Initialize the string and the counter
		self.string = ''
		self.reset()


	def increment(self, amount = 1):
		'''
		Increment the internal counter by a specified amount.
		'''
		if amount + self.value > self.bounds[1]:
			raise ValueError('Cannot increment counter beyond upper bound')

		self.value += amount
		self.makebar()


	def makebar(self):
		'''
		Return a string representing the progress bar.
		'''
		pct = float(self.value - self.bounds[0]) / float(self.range)
		# Figure the number of complete characters in the bar
		nchar = int(self.barwidth * pct)
		string = '#' * nchar + ' ' * (self.barwidth - nchar)

		self.string = '[' + string + '] %5.1f%%' % (100 * pct)


	def reset(self):
		'''
		Reset the bar counter.
		'''
		self.value = self.bounds[0]
		self.makebar()


	def __str__(self): return self.string
