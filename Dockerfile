# Small Debian image with Python 3.13 preinstalled.
FROM python:3.13-slim

# LightGBM's Linux wheel needs the OpenMP runtime (the Linux cousin of libomp on the Mac).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the uv binary from its official image instead of pip-installing it.
COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /bin/uv

# Run as an unprivileged user (uid 1000) instead of root: standard container hygiene.
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# Use the image's Python, precompile .py files for faster startup, copy (not link) packages.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/home/user/app/.venv/bin:$PATH"

# 1) Dependencies only. This layer is cached until pyproject.toml or uv.lock change.
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# 2) Our package, then install it into the same environment.
COPY --chown=user src ./src
RUN uv sync --locked --no-dev

# 3) What the services need at runtime: API and app code, model and panel.
COPY --chown=user api ./api
COPY --chown=user app/streamlit_app.py ./app/streamlit_app.py
COPY --chown=user models/model.txt ./models/model.txt
COPY --chown=user app/data/panel.parquet ./app/data/panel.parquet

# Default command: the FastAPI service on 8000. `make docker-run-app` overrides it to run
# the Streamlit app on 8501 from this same image.
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
