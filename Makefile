.PHONY: install lint test data backtest grouped expanded all

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

grouped:
	fuelsignal grouped

expanded:
	python scripts/fetch_series.py && fuelsignal expanded
