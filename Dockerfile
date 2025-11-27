# Dockerfile para Koyeb com GPU (CUDA 12.4)
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# Instalar Python 3.10 e dependências do sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Criar symlink para python
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/bin/python3.10 /usr/bin/python3

# Copiar requirements primeiro (para cache de layers)
COPY requirements.txt .

# Instalar PyTorch com CUDA 12.4 primeiro
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Instalar outras dependências
RUN pip3 install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Criar diretórios necessários
RUN mkdir -p storage/videos storage/uploads storage/reports storage/snapshots storage/logs

# Variáveis de ambiente padrão
ENV MODEL_PATH=models/best.pt
ENV PORT=8000
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV PYTHONUNBUFFERED=1

# Expor porta
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Comando de start
CMD ["python3", "-m", "uvicorn", "backend.api_optimized:app", "--host", "0.0.0.0", "--port", "8000"]

