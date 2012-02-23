'''
General-purpose, non-numerical routines.
'''

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
		self.range = bounds[1] - bounds[0] + 1
		self.width = width

		# Initialize the string and the counter
		self.string = ''
		self.value = self.bounds[0]


	def increment(self, amount = 1):
		'''
		Increment the internal counter by a specified amount.
		'''
		if amount + self.value > self.bounds[1] + 1:
			raise ValueError('Cannot increment counter beyond upper bound')

		self.value += amount
		self.makebar()


	def makebar(self):
		'''
		Return a string representing the progress bar.
		'''
		pct = float(self.value - self.bounds[0]) / float(self.range)
		# Figure the number of complete characters in the bar
		nchar = int(self.width * pct)
		string = '#' * nchar + ' ' * (self.width - nchar)

		self.string = '[' + string + '] %5.1f%%' % (100 * pct)


	def __str__(self): return self.string
