FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
ENV PORT 8080

WORKDIR /app

# 1. Copy requirements from the CURRENT directory
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the script from the CURRENT directory
COPY . .

# 3. Start the service
# NOTICE THE CHANGE HERE:
# Old: backend_agent.back_agent:app
# New: back_agent:app
CMD exec uvicorn back_agent:app --host 0.0.0.0 --port ${PORT}