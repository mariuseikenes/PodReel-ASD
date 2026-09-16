FROM python:3.11-slim 

RUN apt-get update && apt-get install -y \
  libgl1 libglib2.0-0 ffmpeg \
  && rm -rf /var/lib/apt/lists/* 

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./ 
COPY src ./src
RUN pip install uv && uv sync --frozen --no-dev 


EXPOSE 8000 
CMD ["uv", "run", "uvicorn", "podreel_asd.main:app", "--host", "0.0.0.0", "--port", "8000"]
