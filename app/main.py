import asyncio
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.text_processing import clean_text

app = FastAPI(
    title="STF Text Classification API",
    description="API para classificação de textos jurídicos",
    version="1.0.0",
)


class TextoRequest(BaseModel):
    texto: str


class RamoDireitoResponse(BaseModel):
    id: int
    ramo_direito: List[str]


@app.get("/")
async def root():
    return {
        "message": "Bem-vindo à API de Classificação de Textos Jurídicos",
        "endpoints": {
            "classificar_texto": {
                "path": "/api/pecas/{id}",
                "method": "POST",
                "description": "Utiliza uma modelo de Machine Learning que classifica o texto jurídico e retorna uma lista com ramos do direito.",
            }
        },
    }


async def inferir_ramo_direito(texto: str) -> List[str]:
    # Clean the text before processing
    texto_limpo = clean_text(texto)
    
    # Simulate some async processing (e.g., calling an external API or ML service)
    await asyncio.sleep(0.1)  # Simulate async work
    return ["Direito Civil"]


@app.post("/api/pecas/{id}", response_model=RamoDireitoResponse)
async def inferir_ramo(id: int, request: TextoRequest):
    try:
        ramos = await inferir_ramo_direito(request.texto)
        return RamoDireitoResponse(id=id, ramo_direito=ramos)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
