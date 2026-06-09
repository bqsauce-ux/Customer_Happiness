FROM python:3.11-slim

WORKDIR /app

COPY src/api/ .
COPY src/models/best_model.joblib ./src/models/

RUN pip install -r requirements.txt

EXPOSE 7860


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
