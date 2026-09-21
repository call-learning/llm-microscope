.PHONY: setup setup-all sync run check clean

setup:
	./scripts/setup.sh

setup-all:
	./scripts/setup.sh --all

sync:
	uv sync --extra captum --extra umap

run:
	uv run streamlit run app.py

check:
	uv run python scripts/check_environment.py

clean:
	rm -rf .venv
