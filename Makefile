default_target: all

.EXPORT_ALL_VARIABLES:

PYTHONPATH=.
COVERAGE_FILE=build/.coverage

venv:
	python3 -m venv venv
	venv/bin/python -m pip install --upgrade pip setuptools wheel

install:
	python -m pip install -r requirements.txt

plugin:
	mkdir -p build/plugin && \
	cd build/plugin && \
	cmake ../../watsor/plugin
	$(MAKE) -C build/plugin all
	cp build/plugin/*.so watsor/

test:
	python watsor/test/test_spawn.py

coverage:
	coverage run -m unittest discover -v 2 -s watsor/test
	coverage combine
	coverage report -m

package:
	python setup.py sdist --dist-dir=build/dist bdist_wheel --dist-dir=build/dist

image:
	docker build --tag watsor.base     --file docker/Dockerfile.base     .
	docker build --tag watsor.gpu.base --file docker/Dockerfile.gpu.base .
	docker build --tag watsor.pi3.base --file docker/Dockerfile.pi3.base .
	docker build --tag watsor.pi4.base --file docker/Dockerfile.pi4.base .
	docker build --tag watsor          --file docker/Dockerfile          .
	docker build --tag watsor.gpu      --file docker/Dockerfile.gpu      .
	docker build --tag watsor.pi3      --file docker/Dockerfile.pi3 .
	docker build --tag watsor.pi4      --file docker/Dockerfile.pi4 .

all: plugin test package
