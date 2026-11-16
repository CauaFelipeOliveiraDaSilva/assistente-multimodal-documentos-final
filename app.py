import base64
import json
import re

import fitz  # PyMuPDF
import streamlit as st
from openai import OpenAI

# ---------- 1. Conexão com o Gemini (plano gratuito) ----------
MODEL = "gemini-2.5-flash"
client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=st.secrets["GEMINI_API_KEY"],
)

# ---------- 2. Prompts ----------
PAGE_PROMPT = (
    "Descreva esta página de documento em português. Transcreva o texto, "
    "reproduza tabelas com todos os valores e, para gráficos, liste cada "
    "valor lido (rótulo e número). Não resuma nem invente dados."
)

CHECK_PROMPT = """Abaixo está a descrição de um documento. Identifique os casos em
que um total declarado deve ser igual à soma de componentes (ex.: total de uma
tabela vs. valores mensais de um gráfico do MESMO período). Use somente
componentes do período do total. Números inteiros, sem separador de milhar.
Responda SOMENTE com JSON, neste formato:
[{"descricao": "...", "periodo": "...", "total_declarado": 0,
  "componentes": {"rotulo": 0}}]
Se não houver, responda []."""

SYSTEM = """Você é um assistente de análise de documentos. Responda em português
usando apenas o conteúdo abaixo. Quando citar números, compare com dados
equivalentes em outras partes do documento e aponte divergências explicitamente.

VERIFICAÇÃO AUTOMÁTICA (calculada por código, confiável):
{checks}
Se houver DIVERGÊNCIA acima, informe-a ao usuário com os valores e a diferença.

DOCUMENTO:
{doc}"""


# ---------- 3. Chamada ao modelo, com aviso amigável se estourar a cota ----------
def ask(messages):
    try:
        r = client.chat.completions.create(
            model=MODEL, temperature=0, messages=messages
        )
        return r.choices[0].message.content
    except Exception as e:
        st.error(f"Falha na chamada ao modelo (talvez o limite gratuito): {e}")
        st.stop()


# ---------- 4. Leitura do PDF (visão) ----------
def pdf_to_pngs(pdf_bytes, max_pages=5):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        yield i + 1, page.get_pixmap(dpi=150).tobytes("png")


def describe_page(n, png):
    b64 = base64.b64encode(png).decode()
    texto = ask([{"role": "user", "content": [
        {"type": "text", "text": PAGE_PROMPT},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}])
    return f"### Página {n}\n{texto}"


@st.cache_data(show_spinner=False)
def index_pdf(pdf_bytes):
    return "\n\n".join(describe_page(n, p) for n, p in pdf_to_pngs(pdf_bytes))


# ---------- 5. Verificação cruzada (modelo extrai, Python calcula) ----------
def run_checks(doc):
    texto = ask([{"role": "user",
                  "content": CHECK_PROMPT + "\n\nDOCUMENTO:\n" + doc}])
    m = re.search(r"\[.*\]", texto, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return "Verificação automática indisponível."

    linhas = []
    for it in items:
        try:
            total = it["total_declarado"]
            soma = sum(it["componentes"].values())
        except (KeyError, TypeError, AttributeError):
            continue
        status = "OK" if soma == total else f"DIVERGÊNCIA (diferença de {soma - total})"
        linhas.append(
            f"- {it.get('descricao', '')} ({it.get('periodo', '')}): "
            f"total declarado = {total}, soma dos componentes = {soma} → {status}"
        )
    return "\n".join(linhas) or "Nenhuma verificação encontrada."


# ---------- 6. Interface ----------
st.title("Assistente Multimodal de Documentos")
file = st.file_uploader("Envie um PDF", type="pdf")

if file:
    if st.session_state.get("file_name") != file.name:
        with st.spinner("Lendo páginas, tabelas e gráficos..."):
            st.session_state.doc = index_pdf(file.getvalue())
        with st.spinner("Conferindo totais e somas..."):
            st.session_state.checks = run_checks(st.session_state.doc)
        st.session_state.file_name = file.name
        st.session_state.messages = []

    st.subheader("Verificação automática de consistência")
    st.info(st.session_state.checks)

    for m in st.session_state.messages:
        st.chat_message(m["role"]).write(m["content"])

    if q := st.chat_input("Pergunte sobre o documento"):
        st.session_state.messages.append({"role": "user", "content": q})
        st.chat_message("user").write(q)
        a = ask([
            {"role": "system",
             "content": SYSTEM.format(doc=st.session_state.doc,
                                      checks=st.session_state.checks)},
            *st.session_state.messages,
        ])
        st.session_state.messages.append({"role": "assistant", "content": a})
        st.chat_message("assistant").write(a)
