.PHONY: install lint test data backtest all

install:
	pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest -q

data:
	fuelsignal data

backtest:
	fuelsignal backtest

all: lint test data backtest
