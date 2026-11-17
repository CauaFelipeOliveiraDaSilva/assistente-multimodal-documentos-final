import base64
import json
import re

import fitz  # PyMuPDF
import streamlit as st
from openai import OpenAI

# ---------- 1. Conexão com o Gemini (plano gratuito) ----------
client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=st.secrets["GEMINI_API_KEY"],
)


@st.cache_resource(show_spinner=False)
def escolher_modelo():
    """Pergunta à API quais modelos existem e escolhe um 'flash' com visão."""
    padrao = "gemini-3-flash-preview"
    try:
        ids = [m.id.replace("models/", "") for m in client.models.list()]
    except Exception:
        return padrao
    if padrao in ids:
        return padrao
    excluir = ("lite", "image", "tts", "audio", "live", "embedding", "imagen", "veo")
    flash = sorted(
        (i for i in ids if "flash" in i and not any(x in i for x in excluir)),
        reverse=True,
    )
    return flash[0] if flash else padrao


MODEL = escolher_modelo()
st.sidebar.caption(f"Modelo em uso: {MODEL}")

# Margem aceita quando um valor foi lido pela escala de um gráfico (3%).
# Aumente para tolerar mais erro de leitura, diminua para ser mais rigoroso.
TOLERANCIA_ESTIMADO = 0.03

# ---------- 2. Prompts ----------
PAGE_PROMPT = """Descreva esta página de documento em português.

1. Transcreva o texto e reproduza as tabelas com todos os valores.

2. Para CADA gráfico, escreva um bloco neste formato:
GRÁFICO: <título>
Tipo: <barras, linhas, pizza, empilhado...>
Eixo X: <o que representa> | Eixo Y: <unidade e escala>
Valores: <rótulo = valor> para cada ponto ou barra, sem pular nenhum
Valores impressos no gráfico: <sim ou não>
Período ou escopo coberto: <ex.: janeiro a junho>

Regras para os gráficos:
- Leia os valores SOMENTE pela escala e pela posição visual.
- NÃO use totais citados no texto ou em tabelas para confirmar ou ajustar
  os valores, e NÃO comente divergências: só descreva o que o gráfico mostra.
- Se os valores NÃO estão impressos no gráfico, marque TODOS como (estimado).
- No bloco "Valores", escreva apenas o que vem do próprio gráfico. Nunca copie
  para dentro dele números que só aparecem no texto.
- Em gráficos de pizza, os valores são percentuais estimados pelo tamanho das
  fatias (ex.: metade = 50%, um quarto = 25%).
- Se houver dois eixos Y, indique a qual eixo pertence cada série.

3. Não resuma nem invente dados."""

CHECK_PROMPT = """Abaixo está a descrição de um documento. Faça três tipos de
conferência e responda SOMENTE com JSON, sem texto extra.

1) "somas": casos em que um total declarado deve ser igual à soma de componentes
(ex.: total de tabela vs. valores mensais de um gráfico do MESMO período).
Use somente componentes do período do total. Inclua gráficos de percentuais:
o total declarado é 100 e os componentes são os percentuais lidos do gráfico.

2) "comparacoes": o MESMO item aparecendo em dois lugares DIFERENTES do
documento (ex.: tabela e gráfico) com valores que deveriam ser iguais. Cada
valor deve vir de uma fonte diferente. NUNCA compare um valor do texto com ele
mesmo, nem com um número que foi copiado do texto para dentro da descrição do
gráfico. Liste todos os itens comparáveis, mesmo que os valores sejam iguais.

3) "calculos": afirmações do texto que dependem de uma conta com dados do
documento. Tipos aceitos:
- "variacao_percentual": ex. "cresceu 150% da semana 1 à semana 8"
  (use valor_inicial, valor_final e valor_declarado em %)
- "media": ex. "média de 85" (use a lista valores e valor_declarado)
Não faça a conta: apenas extraia os números usados.

Use números sem separador de milhar e sem o símbolo %. O campo "estimado" é
true se algum valor usado foi marcado como (estimado) na descrição, e false
caso contrário. Formato:
{"somas": [{"descricao": "...", "periodo": "...", "total_declarado": 0,
            "componentes": {"rotulo": 0}, "estimado": false}],
 "comparacoes": [{"item": "...", "fonte_a": "tabela", "valor_a": 0,
                  "fonte_b": "gráfico", "valor_b": 0, "estimado":
