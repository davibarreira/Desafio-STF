## Orientações Gerais

- Crie um repositório no github;
- Compartilhe com o recrutador o link do seu repositório.

## Ambiente de Aplicação

- Utilize qualquer ferramenta ou biblioteca em **Python** para auxilia-lo a resolver o desafio;
- Iremos disponibilizar um dataset contendo:
  - _texto_bruto_: texto de uma peça processual;
  - _ramo_direito_: lista de ramos do direito.

## Desafio

Você foi contratado por um escritório de advocacia para desenvolver uma aplicação que disponibilizará 
uma API REST que tem por objetivo classificar automaticamente textos de peças jurídicas no ramo do direito 
correspondente (e.g., direito civil, penal, trabalhista, etc.).

### API

#### Inferência do ramo do direito

`POST: /api/pecas/{id}`  

Exemplo:
- `POST: /api/pecas/1`

Body
``` json
{"texto": "texto da peça."}
```

Response
``` json
{
    "id": 1,
    "ramo_direito": ["Direito Constitucional", "Direito Administrativo"]
}
```

## O que iremos avaliar

- Arquitetura da solução;
- Funcionamento da aplicação;
- Documentação;
- Cobertura de testes;
- Organização e estrutura do código (boas práticas);
- Modelo.

## Diferenciais

- Uso de Docker;
- Integração Contínua (IC).

## Prazo

7 dias.
