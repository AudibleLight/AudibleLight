.PHONY: install tests docs fix download

install:
	sudo apt update
	sudo apt install -y libsox-dev libsox-fmt-all freeglut3-dev pandoc
	poetry install --no-interaction

tests:
	poetry run flake8 audiblelight --count --select=E9,F63,F7,F82 --show-source --statistics
	poetry run pytest --nbmake -n=1 -v --ignore-glob='*.py' --reruns 3 --reruns-delay 5 notebooks
	poetry run pytest -n 1 -vv --cov-branch --cov-report term-missing --cov-report=xml --cov=audiblelight tests --reruns 3 --reruns-delay 5 --random-order

fix:
	poetry run pre-commit install
	poetry run pre-commit run --all-files

docs:
	cd docs && poetry run make clean
	poetry run sphinx-build docs docs/_build

download:
	poetry run python scripts/download_data/download_fma.py --cleanup
	poetry run python scripts/download_data/download_gibson.py --cleanup
	poetry run python scripts/download_data/download_fsd.py --cleanup