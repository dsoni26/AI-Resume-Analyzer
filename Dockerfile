FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 --retries 10 -r requirements.txt

EXPOSE 8501

CMD streamlit run main.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
