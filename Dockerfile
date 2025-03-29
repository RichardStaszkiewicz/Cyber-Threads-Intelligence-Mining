# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy project files
COPY ./requirements.txt /app/requirements.txt
COPY ./scripts /app/scripts
COPY ./models /app/models
COPY ./datasets /app/datasets
COPY ./config.yaml /app/config.yaml

RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI server port
EXPOSE 8000
EXPOSE 7687

# Run the FastAPI app
CMD ["sleep", "infinity"]
# CMD ["python", "main.py"]
# CMD ["python", "scripts/api.py"]
# CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8000"]
