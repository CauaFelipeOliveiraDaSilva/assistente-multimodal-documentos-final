import base64
import json
import re
import time

import fitz  # PyMuPDF
import streamlit as st
from openai import OpenAI

# ---------- 0. Layout (somente visual) ----------
st.set_page_config(page_title="MW Assist", page_icon="🔷", layout="centered")


def aplicar_tema():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --azul: #2563eb;
    --azul-claro: #60a5fa;
    --preto: #05070d;
    --card: #0d1220;
    --borda: #1e2a44;
    --texto: #e6ecf8;
    --suave: #8fa0c0;
}

html, body, .stApp, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(1100px 560px at 85% -10%, rgba(37, 99, 235, 0.20), transparent 60%),
        var(--preto);
    color: var(--texto);
}

[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

/* Barra lateral */
[data-testid="stSidebar"] {
    background: #070a12;
    border-right: 1px solid var(--borda);
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--suave); }
[data-testid="stSidebar"] h3 { font-size: 1rem; letter-spacing: 0.04em; }

/* Itens da lista de conversas (botões da barra lateral) */
[data-testid="stSidebar"] .stButton > button {
    background: transparent;
    color: var(--texto);
    border: 1px solid var(--borda);
    border-radius: 10px;
    font-weight: 500;
    font-size: 0.85rem;
    justify-content: flex-start;
    text-align: left;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(37, 99, 235, 0.15);
    border-color: var(--azul);
    color: #ffffff;
}

/* Botão "Nova conversa" */
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--azul);
    color: #ffffff;
    border: 0;
    font-weight: 600;
    justify-content: center;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #1d4ed8;
}

/* Detalhes técnicos discretos */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: transparent;
    border: 0;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: var(--suave);
    font-size: 0.78rem;
}

/* Cabeçalho do app */
.mw-hero { padding: 6px 0 22px; }
.mw-chip {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    color: var(--azul-claro);
    border: 1px solid var(--borda);
    background: rgba(37, 99, 235, 0.12);
    padding: 5px 14px;
    border-radius: 999px;
}
.mw-title {
    font-size: 3.4rem;
    font-weight: 800;
    line-height: 1.05;
    margin: 14px 0 8px;
    background: linear-gradient(90deg, #ffffff 0%, #60a5fa 55%, #2563eb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.mw-sub { color: var(--suave); font-size: 1.05rem; }

/* Títulos */
h1, h2, h3, [data-testid="stHeading"] { color: #ffffff; font-weight: 700; }

/* Envio de arquivo */
[data-testid="stFileUploader"] section {
    background: var(--card);
    border: 1.5px dashed var(--azul);
    border-radius: 14px;
}
[data-testid="stFileUploader"] label p { color: var(--texto); font-weight: 600; }

/* Botões */
.stButton > button,
[data-testid="stFileUploader"] button {
    background: var(--azul);
    color: #ffffff;
    border: 0;
    border-radius: 10px;
    font-weight: 600;
}
.stButton > button:hover,
[data-testid="stFileUploader"] button:hover {
    background: #1d4ed8;
    color: #ffffff;
}

/* Quadro de verificação (st.info) e avisos */
[data-testid="stAlert"] {
    background: var(--card);
    border: 1px solid var(--borda);
    border-left: 4px solid var(--azul);
    border-radius: 12px;
    color: var(--texto);
}

/* Texto extraído */
[data-testid="stExpander"] {
    background: var(--card);
    border: 1px solid var(--borda);
    border-radius: 12px;
}

/* Chat */
[data-testid="stChatMessage"] {
    background: var(--card);
    border: 1px solid var(--borda);
    border-radius: 14px;
    padding: 14px;
}
[data-testid="stBottom"] > div { background: var(--preto); }
[data-testid="stChatInput"] > div {
    background: var(--card);
    border: 1px solid var(--azul);
    border-radius: 14px;
}
[data-testid="stChatInput"] textarea { color: var(--texto); }
</style>
        """,
        unsafe_allow_html=True,
    )


aplicar_tema()

# ---------- 1. Conexão com o Gemini (plano gratuito) ----------
client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=st.secrets["GEMINI_API_KEY"],
    timeout=60,       # cada chamada desiste em 60 s em vez de ficar travada
    max_retries=0,    # as novas tentativas são feitas pela função ask()
)


@st.cache_resource(show_spinner=False)
def listar_modelos():
    """Lista modelos Gemini com visão, em ordem de preferência."""
    padrao = "gemini-3-flash-preview"
    try:
        ids = [m.id.replace("models/", "") for m in client.models.list()]
    except Exception:
        return [padrao]
    excluir = ("omni", "image", "tts", "audio", "live", "embedding", "imagen", "veo")
    flash = [i for i in ids if "flash" in i and not any(x in i for x in excluir)]
    principais = sorted((i for i in flash if "lite" not in i), reverse=True)
    leves = sorted((i for i in flash if "lite" in i), reverse=True)
    ordem = ([padrao] if padrao in ids else []) + principais + leves
    return list(dict.fromkeys(ordem)) or [padrao]


MODELOS = listar_modelos()[:4]

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
                  "fonte_b": "gráfico", "valor_b": 0, "estimado": false}],
 "calculos": [{"afirmacao": "...", "tipo": "variacao_percentual",
               "valor_inicial": 0, "valor_final": 0, "valores": [],
               "valor_declarado": 0, "estimado": false}]}
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


# ---------- 3. Chamada ao modelo, com nova tentativa e troca de modelo ----------
ERROS_TEMPORARIOS = ("503", "500", "502", "504", "overloaded",
                     "unavailable", "timed out", "timeout")


def ask(messages):
    erro = None
    for modelo in MODELOS:
        for tentativa in range(3):
            try:
                r = client.chat.completions.create(model=modelo, messages=messages)
                return r.choices[0].message.content
            except Exception as e:
                erro = e
                texto = str(e).lower()
                if "429" in texto and tentativa == 0:
                    st.toast(f"Limite de uso em {modelo}. Esperando 20 s...")
                    time.sleep(20)  # limite por minuto: espera e tenta de novo
                    continue
                if any(t in texto for t in ERROS_TEMPORARIOS) and tentativa < 2:
                    espera = 8 * (tentativa + 1)  # 8 s e depois 16 s
                    st.toast(f"{modelo} sobrecarregado. Nova tentativa em {espera} s...")
                    time.sleep(espera)
                    continue
                break  # cota do dia esgotada ou erro persistente: próximo modelo
        # o erro de cada modelo fica guardado nos "Detalhes técnicos" da barra lateral
        st.session_state.setdefault("falhas", []).append(f"{modelo}: {str(erro)[:150]}")
    st.error(
        "Todos os modelos falharam. Erro 429 = limite de uso gratuito; "
        "erro 503 = servidor do Google sobrecarregado (tente de novo em alguns "
        f"minutos). Último erro: {erro}"
    )
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


def status_de(dif, ref, estimado, folga=0.0):
    """OK / OK dentro da margem / POSSÍVEL DIVERGÊNCIA / DIVERGÊNCIA."""
    d = abs(dif)
    if d <= folga + 1e-9:
        return "OK"
    if estimado:
        if d <= TOLERANCIA_ESTIMADO * abs(ref) + folga:
            return "OK (dentro da margem de leitura do gráfico)"
        return "POSSÍVEL DIVERGÊNCIA (valor estimado pela escala do gráfico)"
    return "DIVERGÊNCIA"


def com_diferenca(status, dif):
    if status.startswith("OK"):
        return status
    return f"{status} — a diferença é {fmtn(abs(dif))}"


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

    # --- somas: total declarado vs. soma dos componentes ---
    for it in dados.get("somas", []):
        try:
            total = num(it["total_declarado"])
            soma = sum(num(v) for v in it["componentes"].values())
        except (KeyError, TypeError, AttributeError, ValueError):
            continue
        dif = soma - total
        status = com_diferenca(status_de(dif, total, eh_estimado(it)), dif)
        linhas.append(
            f"- {it.get('descricao', '')} ({it.get('periodo', '')}): "
            f"total declarado = {fmtn(total)}, soma dos componentes = {fmtn(soma)} → {status}")

    # --- comparações: o mesmo item em duas fontes diferentes ---
    for c in dados.get("comparacoes", []):
        try:
            a, b = num(c["valor_a"]), num(c["valor_b"])
        except (KeyError, TypeError, ValueError):
            continue
        fa, fb = str(c.get("fonte_a", "")), str(c.get("fonte_b", ""))
        if fa.strip().lower() == fb.strip().lower():
            continue  # mesma fonte: comparação sem sentido
        dif = a - b
        status = com_diferenca(status_de(dif, max(abs(a), abs(b)), eh_estimado(c)), dif)
        linhas.append(
            f"- {c.get('item', '')}: {fa or 'fonte A'} = {fmtn(a)} vs. "
            f"{fb or 'fonte B'} = {fmtn(b)} → {status}")

    # --- cálculos: o Python refaz a conta da afirmação ---
    for c in dados.get("calculos", []):
        try:
            declarado = num(c["valor_declarado"])
            if c.get("tipo") == "variacao_percentual":
                ini, fim = num(c["valor_inicial"]), num(c["valor_final"])
                real = (fim - ini) / ini * 100
            elif c.get("tipo") == "media":
                vals = [num(v) for v in c["valores"]]
                real = sum(vals) / len(vals)
            else:
                continue
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        dif = real - declarado
        # folga de 0,5 porque o texto costuma arredondar o resultado
        status = com_diferenca(
            status_de(dif, declarado, eh_estimado(c), folga=0.5), dif)
        linhas.append(
            f"- {c.get('afirmacao', '')}: declarado = {round(declarado, 1)}, "
            f"recalculado = {round(real, 1)} → {status}")

    return "\n".join(linhas) or "Nenhuma verificação encontrada."


@st.cache_data(show_spinner=False)
def checks_pdf(doc):
    """Guarda a verificação em cache: o mesmo documento não gasta cota duas vezes."""
    return run_checks(doc)


# ---------- 6. Conversas (estilo chat de IA) ----------
def nova_conversa():
    """Cria uma conversa vazia, a menos que já exista uma vazia aberta."""
    conversas = st.session_state.conversas
    atual = conversas.get(st.session_state.ativa)
    if atual and atual["doc"] is None and not atual["messages"]:
        return  # já está numa conversa vazia
    st.session_state.proximo_id += 1
    cid = st.session_state.proximo_id
    conversas[cid] = {"nome_pdf": None, "doc": None, "checks": None, "messages": []}
    st.session_state.ativa = cid


def excluir_conversa(cid):
    conversas = st.session_state.conversas
    conversas.pop(cid, None)
    if st.session_state.ativa == cid:
        restantes = list(conversas)
        st.session_state.ativa = restantes[-1] if restantes else None
    if not conversas:
        nova_conversa()


if "conversas" not in st.session_state:
    st.session_state.conversas = {}
    st.session_state.proximo_id = 0
    st.session_state.ativa = None
    nova_conversa()

conversas = st.session_state.conversas

# ---------- 7. Barra lateral ----------
with st.sidebar:
    if st.button("＋  Nova conversa", key="nova", type="primary",
                 use_container_width=True):
        nova_conversa()
        st.rerun()

    st.markdown("### Conversas")
    for cid in reversed(list(conversas)):
        c = conversas[cid]
        titulo = c["nome_pdf"] or "Nova conversa"
        titulo = titulo if len(titulo) <= 22 else titulo[:19] + "..."
        rotulo = ("● " if cid == st.session_state.ativa else "") + titulo
        col_a, col_b = st.columns([5, 1])
        if col_a.button(rotulo, key=f"conv_{cid}", use_container_width=True):
            st.session_state.ativa = cid
            st.rerun()
        if col_b.button("✕", key=f"del_{cid}", help="Excluir conversa"):
            excluir_conversa(cid)
            st.rerun()

    st.write("")
    with st.expander("Detalhes técnicos", expanded=False):
        st.caption("Modelos (em ordem de uso): " + ", ".join(MODELOS))
        for falha in st.session_state.get("falhas", [])[-8:]:
            st.caption("⚠ " + falha)

# ---------- 8. Tela principal ----------
st.markdown(
    """
<div class="mw-hero">
    <span class="mw-chip">ANÁLISE DE DOCUMENTOS</span>
    <div class="mw-title">MW Assist</div>
    <div class="mw-sub">Lê tabelas e gráficos e confere se os números fazem sentido.</div>
</div>
    """,
    unsafe_allow_html=True,
)

atual = conversas[st.session_state.ativa]

if atual["doc"] is None:
    # Conversa vazia: o PDF enviado aqui fica ligado a esta conversa
    file = st.file_uploader("Envie um PDF para começar esta conversa", type="pdf",
                            key=f"up_{st.session_state.ativa}")
    if file is not None:
        with st.spinner("Lendo páginas, tabelas e gráficos..."):
            doc = index_pdf(file.getvalue())
        with st.spinner("Conferindo totais, somas e cálculos..."):
            checks = checks_pdf(doc)
        atual["nome_pdf"] = file.name
        atual["doc"] = doc
        atual["checks"] = checks
        st.rerun()
else:
    st.caption(f"Documento: {atual['nome_pdf']}  ·  para analisar outro PDF, use ＋ Nova conversa")
    st.subheader("Verificação automática de consistência")
    st.info(atual["checks"])

    with st.expander("Texto extraído do documento (leitura do modelo)"):
        st.markdown(atual["doc"])

    for m in atual["messages"]:
        st.chat_message(m["role"]).write(m["content"])

    if q := st.chat_input("Pergunte sobre o documento"):
        atual["messages"].append({"role": "user", "content": q})
        st.chat_message("user").write(q)
        a = ask([
            {"role": "system",
             "content": SYSTEM.format(doc=atual["doc"], checks=atual["checks"])},
            *atual["messages"],
        ])
        atual["messages"].append({"role": "assistant", "content": a})
        st.chat_message("assistant").write(a)
