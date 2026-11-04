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

state-backtest:
	for year in $$(seq 2000 2025); do statedistress dfm-year --year $$year || exit 1; done
	statedistress backtest
