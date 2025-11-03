#!/bin/bash

# Este script é o novo "Manual de Construção" para o Render.

# 1. Parar o script se qualquer comando falhar
set -e

# 2. Atualizar a lista de pacotes do sistema do servidor
echo "Iniciando atualização do sistema..."
apt-get update -y

# 3. Instalar as "ferramentas de montagem" (compiladores) que o PyMC precisa
echo "Instalando compiladores (build-essential, gfortran)..."
apt-get install -y build-essential gfortran

# 4. Agora, finalmente, instalar nossas bibliotecas Python
echo "Instalando dependências do Python..."
pip install -r requirements.txt

echo "Build concluído com sucesso."