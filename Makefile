# Shortcuts for the common project tasks. Run `make <target>`.
# Recipe lines must be indented with a TAB, not spaces.

DATASET := patelris/crop-yield-prediction-dataset
IMAGE   := crop-yield-forecast

.PHONY: data eda train serve app docker-build docker-run lint test

data:  ## Download the Kaggle CSVs into data/raw/
	uv run kaggle datasets download -d $(DATASET) -p data/raw --unzip

eda:  ## Open the EDA notebook
	uv run jupyter lab notebooks/01_eda.ipynb

train:  ## Build panel, train baseline + LightGBM, log to MLflow (M2)
	uv run python -m cropyield.train

serve:  ## Run the FastAPI server locally (M3)
	uv run uvicorn api.main:app --reload --port 8000

app:  ## Run the Streamlit app locally (M4)
	uv run streamlit run app/streamlit_app.py --server.port 7860

docker-build:  ## Build the Docker image (M3)
	docker build -t $(IMAGE) .

docker-run:  ## Run the image, exposing the HF Spaces port (M3/M4)
	docker run --rm -p 7860:7860 $(IMAGE)

lint:
	uv run ruff check .

test:
	uv run pytest
