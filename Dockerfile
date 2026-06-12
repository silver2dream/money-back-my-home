FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .
ENV PYTHONUNBUFFERED=1

# 分析請求最長 ~10 秒,timeout 留裕度;2 worker 對 1~4GB VPS 合理
CMD ["gunicorn", "-w", "2", "-t", "120", "-b", "0.0.0.0:8000", "app:app"]
