FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Main/ ./Main/
COPY Config/settings.yaml ./Config/settings.yaml
COPY README.md .

RUN mkdir -p /app/DataBase /app/Logs /app/InputDoc

EXPOSE 8501

CMD ["streamlit", "run", "Main/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
