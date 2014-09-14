'''
These routines and classes simplify the use of the multiprocessing module.
'''
from multiprocessing import Process, cpu_count

def preferred_process_count():
	'''
	If the CPU count can be determined by the multiprocessing module,
	return the count; otherwise, return 1.
	'''
	try: nprocs = cpu_count()
	except NotImplementedError: nprocs = 1
	return nprocs


class ProcessPool(object):
	'''
	This provides a context manager interface that manages a list of
	processes, allowing the caller to attach an arbitrary number of
	processes with arbitrary targets. A convenience function allows all of
	these processes to be joined until completion.

	On leaving the context, terminate() will be called on each process and
	they will be rejoined in succession to ensure a clean exit.
	'''

	def __init__(self):
		'''
		Initialize the thread pool.
		'''
		self._procs = []
	

	def addtask(self, **kwargs):
		'''
		Create a new process, record it in the process list, and pass
		kwargs to multiprocessing.Process.
		'''
		p = Process(**kwargs)
		self._procs.append(p)


	def start(self):
		'''
		Start all processes in the pool.
		'''
		for p in self._procs:
			p.start()


	def wait(self):
		'''
		Repeatedly join each process until all processes have finished.
		Each join has a 1-second timeout. This allows the process table
		to be cleaned as each process dies instead of leaving zombies
		until all are dead.

		Each process is removed from the pool as it dies.
		'''
		while len(self._procs):
			livingprocs = []
			for p in self._procs:
				p.join(1.0)
				if p.is_alive():
					livingprocs.append(p)
			# If the list of living processes shrunk, replace
			# the pool's process list with only those living
			if len(livingprocs) != len(self._procs):
				self._procs = livingprocs


	def __enter__(self):
		'''
		Return self so the object acts as a context manager.
		'''
		return self


	def __exit__(self, exc_type, exc_val, exc_tb):
		'''
		If there are any processes remaining in the pool, terminate and
		join the processes before returning.
		'''
		clean = (exc_type is None and exc_val is None and exc_tb is None)

		try:
			for p in self._procs:
				if p.is_alive():
					p.terminate()
					p.join()
		except TypeError: pass

		self._procs = []

		return clean
