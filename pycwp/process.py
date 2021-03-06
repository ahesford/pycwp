'''
These routines and classes simplify the use of the multiprocessing module.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

from multiprocessing import Process, cpu_count

def preferred_process_count():
	'''
	If the CPU count can be determined by the multiprocessing module,
	return the count; otherwise, return 1.
	'''
	try: nprocs = cpu_count()
	except NotImplementedError: nprocs = 1
	return nprocs


def procname(procnum, funcname=None, hostname=None):
	'''
	Return a string of the form <hostname>-<funcname>-Proc<procnum>, where
	procnum is an integer and the other two components are strings. If
	funcname is None, the basname of sys.argv[0] is used. If hostname is
	None, the output of socket.gethostname() is used.
	'''
	import sys, os
	from socket import gethostname

	if funcname is None: funcname = os.path.basename(sys.argv[0])
	if hostname is None: hostname = gethostname()

	return '{:s}-{:s}-Rank{:d}'.format(hostname, funcname, procnum)


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


	@property
	def unjoined(self):
		'''
		Return the number of processes that have not yet terminated and
		been joined.

		This will be nonzero from the moment addtask() is called until
		self.wait is called and successfully joins every process. (A
		call to join with a limit of None may leave some processes
		unjoined, which will leave this property nonzero.)
		'''
		return len(self._procs)


	def wait(self, timeout=0.1, limit=None):
		'''
		Repeatedly join each process until all processes have finished.
		Each join has the specified timeout, which must be a valid
		argument to multiprocessing.Process.join.

		If limit is not None, it indicates the maximum number of times
		per process that a join will be allowed to timeout. If limit is
		None, each timed-out join will be retried until a successful
		join has been performed.

		Calling wait allows the process table to be cleaned as each
		process dies instead of leaving zombies until all are dead.

		Each process is removed from the pool as it dies.
		'''
		count = 0
		while len(self._procs):
			livingprocs = []
			for p in self._procs:
				p.join(timeout)
				if p.is_alive():
					livingprocs.append(p)
			# If the list of living processes shrunk, replace
			# the pool's process list with only those living
			if len(livingprocs) != len(self._procs):
				self._procs = livingprocs

			# Check to see if the join limit has been reached
			if limit is not None:
				count += 1
				if count >= limit: break


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
