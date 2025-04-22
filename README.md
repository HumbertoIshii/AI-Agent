# YouTube Sentiment Agent

Este projeto é um agente inteligente construído com base no [template do curso de agentes da Hugging Face](https://huggingface.co/spaces/agents-course/First_agent_template), que realiza scraping de comentários de vídeos do YouTube e aplica análise de sentimentos nesses comentários. O agente utiliza ferramentas customizadas para executar essas tarefas e roda localmente com um modelo de linguagem via [Ollama](https://ollama.com/).

## Funcionalidades

- Realiza scraping dos comentários de um vídeo do YouTube a partir da URL
- Aplica análise de sentimentos (positivo, neutro, negativo)
- Retorna um resumo das opiniões nos comentários
- Utiliza ferramentas integradas ao framework de agentes da Hugging Face
- Executa localmente com modelos de linguagem através do Ollama

## Instalação

### 1. Crie um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```


### 3. Instale o Ollama e baixe um modelo local

Você pode [baixar o Ollama aqui](https://ollama.com/download) para executar modelos localmente.

```bash
ollama pull qwen2.5:3b
```

### 4. Execute o agente
```bash
python app.py
```