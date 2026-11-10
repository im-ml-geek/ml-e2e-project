FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY predict.py lead_conversion_model.joblib ./

CMD ["python", "-m", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
