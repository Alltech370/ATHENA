#!/bin/bash
set -e

echo "🔧 Instalando PyTorch com CUDA 12.4 (mais recente)..."
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "📦 Instalando outras dependências..."
pip install --no-cache-dir -r requirements.txt

echo "✅ Build concluído!"

