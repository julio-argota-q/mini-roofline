# Makefile in the project root
.PHONY: check
check:
	uv run python -m miniroofline.cost_model.flops
	uv run python -m miniroofline.cost_model.memory
	uv run python -m miniroofline.cost_model.roofline

.PHONY: f_peak
f_peak:
	uv run python -c "from miniroofline.cost_model.hardware import benchmark_flops; benchmark_flops()"

.PHONY: m_peak
m_peak:
	uv run python -c "from miniroofline.cost_model.hardware import benchmark_memory; benchmark_memory()"

.PHONY: test
test:
	uv run pytest tests/ -v

.PHONY: flops
flops:
	uv run python -m miniroofline.cost_model.flops

# Usefull

.PHONY: format
format:
	uv run ruff format src tests

.PHONY: lint
lint:
	uv run ruff check src tests

.PHONY: exp1
exp1:
	uv run python experiments/exp1_baseline.py

.PHONY: exp2
exp2:
	uv run python experiments/exp2_validation.py

.PHONY: exp3
exp3:
	uv run python experiments/exp3_shape_sensitivity.py

.PHONY: exp4
exp4:
	uv run python experiments/exp4_calibrated_roofline.py	

.PHONY: figs
figs:
	uv run python notebooks/make_figures.py	

.PHONY: report
report:
	uv run quarto render report.qmd --to pdf

.PHONY: clean
clean:
	rm -rf experiments/results/*.json
	rm -rf __pycache__ .pytest_cache

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  check    - run cost model sanity checks"
	@echo "  f_peak   - measure peak FLOP/s on this machine"
	@echo "  m_peak   - measure memory peak GB/s on this machine"
	@echo "  test     - run pytest"
	@echo "  exp1-4   - run experiment scripts"
	@echo "  report   - render the Quarto report to PDF"
	@echo "  clean    - remove generated files"