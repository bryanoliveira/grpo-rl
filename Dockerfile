# docker build -t grpo .
# docker run -e WANDB_API_KEY=<key> grpo

FROM python:3.10.16-slim

# Install system dependencies, including X11 and OpenGL libraries for gymnasium/mujoco rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    bash \
    libgl1-mesa-glx \
    libglfw3 \
    libglew2.2 \
    libosmesa6 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libxi6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Set workdir
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies (including optional mujoco extras)
RUN uv sync && uv sync --extra mujoco

# Make experiment.sh executable
RUN chmod +x launch_all_cpus.sh

# Default command
CMD ["bash", "launch_all_cpus.sh"]
