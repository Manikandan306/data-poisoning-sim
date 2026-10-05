FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generate_data.py retrain_pipeline.py eval.py ./

# eval.py exits with code 1 when the quality gate fails (expected here)
CMD ["sh", "-c", "python generate_data.py && python retrain_pipeline.py && python eval.py"]
