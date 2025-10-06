# Use a stable Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy dependencies first
COPY requirements.txt .

# Always upgrade pip + setuptools + wheel before installing packages
RUN pip install --upgrade pip setuptools wheel

# Install project dependencies
RUN pip install -r requirements.txt

# Copy the rest of your app
COPY . .

# Run your app
CMD ["python", "main.py"]
