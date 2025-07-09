FROM 098824477125.dkr.ecr.us-east-1.amazonaws.com/cardio-ai-predictor:latest

WORKDIR /app
ENV PYTHONPATH=/app
ENV NVIDIA_VISIBLE_DEVICES=none
ENV CUDA_VISIBLE_DEVICES=""
ENV API_BASE_URL="http://localhost"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir xgboost-cpu

COPY . .

EXPOSE 80 8501

CMD ["bash", "-c", "uvicorn src.api:app --host 0.0.0.0 --port=80 & streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]
