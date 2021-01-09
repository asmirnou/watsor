default_target: all

.EXPORT_ALL_VARIABLES:

PYTHONPATH=.

COVERAGE_FILE=build/.coverage

VERSION=$(shell git describe --tags)

DOCKER_ARGS=--tag smirnou/watsor:latest --tag smirnou/watsor:${VERSION}

define release_tags
    $(subst :,$(1):,$(DOCKER_ARGS))
endef

venv:
	python3 -m venv venv
	venv/bin/python -m pip install --upgrade pip setuptools wheel

install:
	python -m pip install -r requirements.txt

test:
	python watsor/test/test_spawn.py

coverage:
	coverage run -m unittest discover -v 2 -s watsor/test
	coverage combine
	coverage report -m

package:
	python setup.py sdist --dist-dir=build/dist bdist_wheel --dist-dir=build/dist

image:
	docker build        --tag watsor.base            --file docker/Dockerfile.base        .
	docker build        --tag watsor.gpu.base        --file docker/Dockerfile.gpu.base    .
	docker buildx build --tag watsor.jetson.base     --file docker/Dockerfile.jetson.base . --platform linux/arm64
	docker build        --tag watsor.pi3.base        --file docker/Dockerfile.pi3.base    .
	docker build        --tag watsor.pi4.base        --file docker/Dockerfile.pi4.base    .
	docker build        $(call release_tags,)        --file docker/Dockerfile             .
	docker build        $(call release_tags,.gpu)    --file docker/Dockerfile.gpu         .
	docker buildx build $(call release_tags,.jetson) --file docker/Dockerfile.jetson      . --platform linux/arm64
	docker build        $(call release_tags,.pi3)    --file docker/Dockerfile.pi3         .
	docker build        $(call release_tags,.pi4)    --file docker/Dockerfile.pi4         .

all: test package
