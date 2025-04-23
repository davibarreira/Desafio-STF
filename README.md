# API - Classificando Textos Jurídicos

API para classificação de textos jurídicos em ramos do direito.

## Introdução
Essa API foi desenvolvida para classificar textos jurídicos em diferentes ramos do direito brasileiro,
através do uso de técnicas de processamento de linguagem natural (NLP) e aprendizado de máquina.
A solução implementa um modelo de classificação multi-label, permitindo que um texto seja classificado em mais de um ramo do direito quando necessário.

## :rocket: Começando

### Pré-requisitos

- Python 3.11+
- uv (gerenciador de pacotes Python)

### Instalação

#### Método 1: Usando uv (recomendado)
```bash
# Instalar uv se ainda não tiver
pip install uv

# Instalar dependências para rodar aplicativo
uv sync

# Instalar dependências de desenvolvimento
uv sync --extra dev
```

#### Método 2: Usando Docker
```bash
# Construir a imagem
docker build -t legal-text-classifier .

# Executar o container
docker run -d -p 8000:8000 --name legal-text-api legal-text-classifier
```

## :memo: Como Usar

### Executando a API

#### Localmente
```bash
# Opção 1: Usando uvicorn (modo desenvolvimento)
uv run uvicorn app.main:app --reload

# Opção 2: Usando uv (modo produção)
uv run python -m app.main
```
Note que colocando o prefixo `uv run`, garantimos que estamos utilizando
a versão de python do projeto.

#### Via Docker
```bash
# Iniciar
docker build -t legal-text-classifier .
docker run -d -p 8000:8000 --name legal-text-api legal-text-classifier
```
Com isso, o container estará rodando a API e já pode ser acessado.

Uma vez que não se quer mais utilizar aquele container, pode deletá-lo usando:
```bash
# Parar
docker stop legal-text-api

# Remover
docker rm legal-text-api
```

### Endpoints Disponíveis

- `GET /`: Documentação da API
- `POST /api/pecas/{id}`: Classificar um texto jurídico

### Exemplo de Uso

```bash
# Testar
curl -s "http://localhost:8000/"
```
Retorna:
```
{"message":"Bem-vindo à API de Classificação de Textos Jurídicos","endpoints":{"classificar_texto":{"path":"/api/pecas/{id}","method":"POST","description":"Utiliza uma modelo de Machine Learning que classifica o texto jurídico e retorna uma lista com ramos do direito."}}}
```

Para obter predições se usa:
```bash
 curl -X POST "http://localhost:8000/api/pecas/1" -H "Content-Type: application/json" -d '{"texto": "EDUCACAO E MUITO IMPORTANTE PROFESSOR EU SEI"}'
```

Resposta esperada:
```json
{"id":1,"ramo_direito":["DIREITO ADMINISTRATIVO E OUTRAS MATÉRIAS DE DIREITO PÚBLICO"]}%                                                                   
```

## :building_construction: Estrutura do Projeto

```
.
├── app/
│   ├── main.py                # API principal
│   ├── text_processing/       # Módulos de processamento de texto
│   │   ├── cleaner.py         # Funções para limpeza de texto
│   │   ├── cleaner_batch.py   # Funções para limpeza em batch
│   │   └── text_processing.py # Script para preprocessamento do dataset
│   └── modelling/            # Módulos para treinamento e predição
│       ├── train.py          # Script de treinamento do modelo
│       └── predict.py        # Funções para fazer predições
├── data/
│   ├── 1_raw/                # Dados brutos
│   └── 2_pro/                # Dados processados
├── models/                   # Modelos treinados
└── tests/                    # Testes automatizados
```

## :wrench: Desenvolvimento

### Pré-processamento dos Dados
Para processar os dados brutos e preparar para treinamento:
```bash
uv run python -m app.text_processing.text_processing
```

### Seleção de Modelo
Para comparar diferentes modelos e selecionar o melhor com base na métrica f1_samples:
```bash
uv run python -m app.modelling.model_selection
```
Este script testa vários algoritmos (Regressão Logística, Random Forest, SVM) com diferentes hiperparâmetros e salva a melhor configuração para ser usada no treinamento final.

### Treinamento do Modelo
Para treinar o modelo de classificação usando a melhor configuração encontrada:
```bash
uv run python -m app.modelling.train
```

### Testes
Para executar os testes e obter o percentual de cobertura:
```bash
uv run pytest --cov=app
```

## :notebook: Notas Adicionais

- Os dados não estão no respositório, pois seu tamanho é maior que o limite do GitHub.
- Os dados originais `dataset_desafio_ramo_direito (1).parquet` recebidos no desafio estão salvos em `data/2_pro/`.
- Os dados processados são salvos em `data/2_pro/`.
- Os modelos treinados são salvos em `models/`.
- A API está disponível em `http://localhost:8000` por padrão.
