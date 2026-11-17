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

# ---------- 2. Prompts ----------
PAGE_PROMPT = """Descreva esta página de documento em português.

1. Transcreva o texto e reproduza as tabelas com todos os valores.

2. Para CADA gráfico, escreva um bloco neste formato:
GRÁFICO: <título>
Tipo: <barras, linhas, pizza, empilhado...>
Eixo X: <o que representa> | Eixo Y: <unidade e escala>
Valores: <rótulo = valor> para cada ponto ou barra, sem pular nenhum
Valores impressos no gráfico: <sim ou não>. Se não, marque cada valor
lido pela escala como (estimado).
Período ou escopo coberto: <ex.: janeiro a junho>

3. Não resuma nem invente dados."""

CHECK_PROMPT = """Abaixo está a descrição de um documento. Faça dois tipos de
conferência e responda SOMENTE com JSON, sem texto extra.

1) "somas": casos em que um total declarado deve ser igual à soma de componentes
(ex.: total de tabela vs. valores mensais de um gráfico do MESMO período).
Use somente componentes do período do total. Inclua gráficos de percentuais:
o total declarado é 100 e os componentes são os percentuais lidos do gráfico.

2) "comparacoes": o MESMO item aparecendo em dois lugares do documento (ex.:
tabela e gráfico) com valores que deveriam ser iguais. Liste TODOS os itens
comparáveis, mesmo que os valores sejam iguais.

Use números sem separador de milhar e sem o símbolo %. O campo "estimado" é
true se algum valor usado foi marcado como (estimado) na descrição, e false
caso contrário. Formato:
{"somas": [{"descricao": "...", "periodo": "...", "total_declarado": 0,
            "componentes": {"rotulo": 0}, "estimado": false}],
 "comparacoes": [{"item": "...", "fonte_a": "tabela", "valor_a": 0,
                  "fonte_b": "gráfico", "valor_b": 0, "estimado": false}]}
Se não houver, use listas vazias."""

SYSTEM = """Você é um assistente de análise de documentos. Responda em português
usando apenas o conteúdo abaixo. Quando citar números, compare com dados
equivalentes em outras partes do documento e aponte divergências explicitamente.
Valores marcados como (estimado) foram lidos pela escala de um gráfico e têm
margem de erro: avise o usuário quando usá-los.

VERIFICAÇÃO AUTOMÁTICA (calculada por código, confiável):
{checks}
Se houver DIVERGÊNCIA acima, informe-a ao usuário com os valores e a diferença.

DOCUMENTO:
{doc}"""


# ---------- 3. Chamada ao modelo, com aviso amigável se der erro ----------
def ask(messages):
    try:
        r = client.chat.completions.create(model=MODEL, messages=messages)
        return r.choices[0].message.content
    except Exception as e:
        st.error(f"Falha na chamada ao modelo '{MODEL}' (nome do modelo ou limite gratuito): {e}")
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
def num(x):
    """Converte o valor devolvido pelo modelo (número ou texto) em número."""
    if isinstance(x, (int, float)):
        return x
    s = re.sub(r"[^\d,.\-]", "", str(x))
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    return float(s)


def fmtn(x):
    return int(x) if x == int(x) else round(x, 2)


def eh_estimado(item):
    return item.get("estimado") is True or str(item.get("estimado")).lower() == "true"


def status_de(dif, estimado):
    if abs(dif) < 1e-9:
        return "OK"
    if estimado:
        return "POSSÍVEL DIVERGÊNCIA (valor estimado pela escala do gráfico)"
    return "DIVERGÊNCIA"


def run_checks(doc):
    texto = ask([{"role": "user",
                  "content": CHECK_PROMPT + "\n\nDOCUMENTO:\n" + doc}])
    m = re.search(r"\{.*\}|\[.*\]", texto, re.S)
    try:
        dados = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return "Verificação automática indisponível."
    if isinstance(dados, list):  # formato antigo
        dados = {"somas": dados}

    linhas = []

    for it in dados.get("somas", []):
        try:
            total = num(it["total_declarado"])
            soma = sum(num(v) for v in it["componentes"].values())
        except (KeyError, TypeError, AttributeError, ValueError):
            continue
        dif = soma - total
        status = status_de(dif, eh_estimado(it))
        if status != "OK":
            status += f" — a soma é {'maior' if dif > 0 else 'menor'} em {fmtn(abs(dif))}"
        linhas.append(
            f"- {it.get('descricao', '')} ({it.get('periodo', '')}): "
            f"total declarado = {fmtn(total)}, soma dos componentes = {fmtn(soma)} → {status}")

    for c in dados.get("comparacoes", []):
        try:
            a, b = num(c["valor_a"]), num(c["valor_b"])
        except (KeyError, TypeError, ValueError):
            continue
        status = status_de(a - b, eh_estimado(c))
        linhas.append(
            f"- {c.get('item', '')}: {c.get('fonte_a', 'fonte A')} = {fmtn(a)} vs. "
            f"{c.get('fonte_b', 'fonte B')} = {fmtn(b)} → {status}")

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

    with st.expander("Texto extraído do documento (leitura do modelo)"):
        st.markdown(st.session_state.doc)

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
