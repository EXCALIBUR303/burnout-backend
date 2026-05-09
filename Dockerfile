# Burnout/AI backend — explicit Dockerfile so Railway picks this up instead
# of nixpacks. Nixpacks was ignoring our aptPkgs config and silently failing
# to install libgomp1 (required by LightGBM in the v4/v5 stacking ensemble).
#
# When this image is detected, Railway uses it instead of nixpacks.

FROM python:3.12-slim
# Why 3.12 instead of 3.13: the slim image for 3.13 doesn't ship pre-built
# wheels for numpy 2.0.2, pandas 2.3.3, etc., and slim images lack the
# compilers to build from source — pip fails with "metadata-generation-failed
# (numpy)". Python 3.12 has wheels for everything in requirements.txt.

# System deps:
#   libgomp1 — GNU OpenMP runtime (required by LightGBM and XGBoost C extensions)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so the layer caches when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend
COPY . .

ENV PYTHONUNBUFFERED=1

# Railway provides $PORT at runtime. We MUST use explicit `sh -c` exec form
# so the variable is expanded by the shell (not passed as literal "$PORT" to
# uvicorn). Shell-form CMD on Railway's Docker runner had been passing the
# raw string -> uvicorn errored with: "Invalid value for '--port': '$PORT'".
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
