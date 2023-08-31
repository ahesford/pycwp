PYTHON=python3
PACKAGE=pycwp

CYTOOLS= $(PACKAGE)/cytools

CYTHON_SRC= $(CYTOOLS)/boxer.c $(CYTOOLS)/eikonal.c \
	    $(CYTOOLS)/interpolator.c $(CYTOOLS)/ptutils.c \
	    $(CYTOOLS)/quadrature.c $(CYTOOLS)/regularize.c \
	    $(CYTOOLS)/stats.c

CYTHON_OPTS= -X language_level=2 -X embedsignature=True

default: install

install: wheel
	mkdir -p build
	TMPDIR=build $(PYTHON) -m pip install \
		--no-deps --no-clean --no-build-isolation \
		--force-reinstall dist/$(PACKAGE)-*-*-*-*.whl

wheel:
	$(PYTHON) -m build --no-isolation --wheel .

sdist:
	$(PYTHON) -m build --no-isolation --sdist .

clean:
	rm -rf build dist
	rm -rf $(PACKAGE).egg-info
	rm -f $(CYTHON_SRC) $(CYTHON_SRC:.c=.html)

cythonize: $(CYTHON_SRC)

cyhtml: $(CYTHON_SRC:.c=.html)

.SUFFIXES: .html .pyx .c
.pyx.html:
	cythonize $(CYTHON_OPTS) -a $<

.pyx.c:
	cythonize $(CYTHON_OPTS) $<
