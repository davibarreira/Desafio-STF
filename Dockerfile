FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY . .

# Install package dependencies
RUN uv sync --extra dev

# Train the model (the model is already trained, so this is not needed, but it's here for reference)
# RUN python -m app.models.train

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "python", "app/main.py"]