FROM python:3.11-slim AS base
WORKDIR /app

COPY requirements.txt app.py ./
RUN pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8001"]