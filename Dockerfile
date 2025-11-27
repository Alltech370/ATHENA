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

# Criar symlink para python (python3 já existe, só criar python)
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Copiar requirements primeiro (para cache de layers)
COPY requirements.txt .

# Instalar PyTorch com CUDA 12.4 primeiro
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Instalar outras dependências
RUN pip3 install --no-cache-dir -r requirements.txt

# Copiar código da aplicação (usando .dockerignore para excluir arquivos desnecessários)
# Copiar apenas o que é necessário para produção
COPY backend/ ./backend/
COPY core/ ./core/
COPY frontend/ ./frontend/
COPY requirements.txt .

# Copiar modelos - tentar ambos os caminhos possíveis
COPY models/ ./models/
# Copiar modelo do caminho padrão (fallback do código)
RUN mkdir -p athena_training_2phase_optimized/models/phase1_complete/athena_phase1_tesla_t4/weights
COPY athena_training_2phase_optimized/models/phase1_complete/athena_phase1_tesla_t4/weights/best.pt ./athena_training_2phase_optimized/models/phase1_complete/athena_phase1_tesla_t4/weights/best.pt

# Criar diretórios necessários
RUN mkdir -p storage/reports storage/snapshots storage/logs \
    snapshots reports logs

# Variáveis de ambiente padrão
# MODEL_PATH será definido pela variável de ambiente no Koyeb ou usa o padrão do código
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

