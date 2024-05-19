FROM python:3.10-slim-buster

# Install dependencies
RUN apt-get update && apt-get install -y python3-venv

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install in virtual environment
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the application code
COPY generator.py /app/generator.py

# Set working directory
WORKDIR /app

#CMD ["python", "generator.py"]
ENTRYPOINT [ "sleep", "infinity" ]