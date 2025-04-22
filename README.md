# API - Classificando Textos Jurídicos

API REST para classificação de textos de peças jurídicas no ramo do direito 
correspondente (e.g., direito civil, penal, trabalhista, etc.).

## Estrutura do Projeto
### Requisitos

- Python 3.11+
- uv (gerenciador de pacotes Python)

### Instalação usando uv (recomendado)
```bash
# Instalar uv se ainda não tiver
pip install uv

# Instalar dependências para rodar aplicativo
uv sync

# Instalar dependências de desensenvolvimento
uv sync --extra dev
```

### Usando Docker

Primeiro vamos construir nosso container:
```bash
docker build -t legal-text-classifier .
```

Em seguinda, vamos rodá-lo com o nome `legal-text-api`:
```bash
docker run -d -p 8000:8000 --name legal-text-api legal-text-classifier
```

Uma vez feito isso, a nossa API estará rodando. Podemos testar usando:
```bash
curl -s "http://localhost:8000/"
```
Isso irá mostrar a documentação da API. Em seguida, para usar o modelo,
envie um POST:
```bash
curl -X POST "http://localhost:8000/api/pecas/1" -H "Content-Type: application/json" -d '{"texto": "AGRAVO EM RECURSO EXTRAORDINÁRIO. DIREITO CONSTITUCIONAL. DIREITO ADMINISTRATIVO. CONCURSO PÚBLICO. POLÍCIA MILITAR. ALTURA MÍNIMA. LEGALIDADE."}'
```

Terminado o uso, podemos parar a aplicação com:
```bash
docker stop legal-text-api
```
Finalmente, se não precisar mais do container, delete usando:
```bash
docker rm legal-text-api
```
