# Use an official CUDA runtime as a parent image
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VERSION=1.1.13 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

# Set work directory
WORKDIR $PYSETUP_PATH

# Install system dependencies
RUN apt-get update \
    && apt-get install -y python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip3 install "poetry==$POETRY_VERSION"

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Copy only requirements to cache them in docker layer
COPY poetry.lock pyproject.toml ./

# Install project dependencies
RUN poetry install --no-dev

# Copy project
COPY . .

# Run tests when the container launches
CMD ["make", "test"]
