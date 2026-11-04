.PHONY: install lint test data backtest grouped expanded state-backtest all

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
state-backtest:
	for year in $$(seq 2000 2025); do statedistress dfm-year --year $$year || exit 1; done
	statedistress backtest
