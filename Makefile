# Shortcuts for the common project tasks. Run `make <target>`.
# Recipe lines must be indented with a TAB, not spaces.

DATASET := patelris/crop-yield-prediction-dataset
IMAGE   := crop-yield-forecast

.PHONY: data eda train serve app docker-build docker-run docker-run-app lint test

data:  ## Download the Kaggle CSVs into data/raw/
	uv run kaggle datasets download -d $(DATASET) -p data/raw --unzip

eda:  ## Open the EDA notebook
	uv run jupyter lab notebooks/01_eda.ipynb

train:  ## Build panel, train baseline + LightGBM, log to MLflow (M2)
	uv run python -m cropyield.train

serve:  ## Run the FastAPI server locally (M3)
	uv run uvicorn api.main:app --reload --port 8000

app:  ## Run the Streamlit app locally on http://localhost:8501 (M4)
	uv run streamlit run app/streamlit_app.py --server.port 8501

docker-build:  ## Build the Docker image (M3)
	docker build -t $(IMAGE) .

docker-run:  ## Run the API container on http://localhost:8000 (M3)
	docker run --rm -p 8000:8000 $(IMAGE)

docker-run-app:  ## Run the Streamlit app container on http://localhost:8501 (M4), same image
	docker run --rm -p 8501:8501 $(IMAGE) \
		streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 \
		--server.headless true --browser.gatherUsageStats false

lint:
	uv run ruff check .

test:
	uv run pytest
