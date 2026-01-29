# Use Python 3.8, to match your local env
FROM docker.artifactory.iav.com/python:3.8.20

# Prevent Python from buffering stdout/stderr (nicer logs)
ENV PYTHONUNBUFFERED=1

# Workdir inside the container
WORKDIR /app

# # System deps (optional; useful if any wheel needs build tools)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (for better caching)
# If you don't have requirements.txt, comment this block out and add it later.
# COPY requirements.txt .

# Copy the rest of the project (including run_matching.py, osc2_parser, scenario_matching, etc.)
COPY . .
RUN pip install --no-cache-dir -r requirements.txt --index-url=https://artifactory.iav.com/artifactory/api/pypi/pypi-remote/simple

# Make sure the project root is importable (osc2_parser, scenario_matching, ...)
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command: run the matcher
ENTRYPOINT ["python", "run_matching.py"]