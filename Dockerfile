FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Default task is 'easy'; override with MY_ENV_V4_TASK env var
ENV MY_ENV_V4_TASK=easy

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
