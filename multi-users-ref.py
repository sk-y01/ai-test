"""멀티유저/멀티세션 RAG 챗봇 - Supabase 기반 사용자·세션·벡터 관리."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client, Client

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "image.png"
LOG_DIR = REPO_ROOT / "logs"

load_dotenv(dotenv_path=ENV_PATH)


def _get_secret(key: str) -> str:
    """st.secrets → os.getenv 우선순위로 시크릿 로드."""
    try:
        val = st.secrets.get(key, "")
        if val:
            return str(val).strip()
    except Exception:
        pass
    return os.getenv(key, "").strip()


OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_ANON_KEY = _get_secret("SUPABASE_ANON_KEY")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_name = f"multi_users_rag_{datetime.now().strftime('%Y%m%d')}.log"
        log_path = LOG_DIR / log_name
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.WARNING)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except (PermissionError, OSError):
        pass

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    for name in (
        "httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("multi_users_rag")


logger = _setup_logging()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""


def remove_separators(text: str) -> str:
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# ---------------------------------------------------------------------------
# Password hashing (SHA-256 + salt)
# ---------------------------------------------------------------------------
def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """SHA-256 해싱. (hash_hex, salt) 반환."""
    if salt is None:
        salt = uuid.uuid4().hex[:16]
    h = hashlib.sha256(f"{salt}{password}".encode("utf-8")).hexdigest()
    return h, salt


def _verify_password(password: str, stored_hash: str) -> bool:
    """stored_hash = 'salt:hash' 형식 검증."""
    if ":" not in stored_hash:
        return False
    salt, expected = stored_hash.split(":", 1)
    h, _ = _hash_password(password, salt)
    return h == expected


# ---------------------------------------------------------------------------
# Supabase Client
# ---------------------------------------------------------------------------
@st.cache_resource
def get_supabase_client() -> Client | None:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


# ---------------------------------------------------------------------------
# User Management
# ---------------------------------------------------------------------------
def register_user(sb: Client, login_id: str, password: str, display_name: str = "") -> dict | None:
    existing = sb.table("users").select("id").eq("login_id", login_id).execute()
    if existing.data:
        return None  # already exists

    h, salt = _hash_password(password)
    password_hash = f"{salt}:{h}"
    row = {
        "login_id": login_id,
        "password_hash": password_hash,
        "display_name": display_name or login_id,
    }
    result = sb.table("users").insert(row).execute()
    return result.data[0] if result.data else None


def authenticate_user(sb: Client, login_id: str, password: str) -> dict | None:
    result = sb.table("users").select("*").eq("login_id", login_id).execute()
    if not result.data:
        return None
    user = result.data[0]
    if _verify_password(password, user["password_hash"]):
        return user
    return None


# ---------------------------------------------------------------------------
# Session Management (user_id 필터 추가)
# ---------------------------------------------------------------------------
def load_all_sessions(sb: Client, user_id: str) -> list[dict]:
    result = (
        sb.table("chat_sessions")
        .select("id, title, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return result.data or []


def save_session_to_db(
    sb: Client, user_id: str, session_id: str, title: str, messages: list[dict],
) -> None:
    existing = (
        sb.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .execute()
    )

    if existing.data:
        sb.table("chat_sessions").update({
            "title": title,
            "updated_at": datetime.now().isoformat(),
        }).eq("id", session_id).eq("user_id", user_id).execute()
        sb.table("chat_messages").delete().eq("session_id", session_id).eq("user_id", user_id).execute()
    else:
        sb.table("chat_sessions").insert({
            "id": session_id,
            "user_id": user_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }).execute()

    if messages:
        batch_size = 50
        for i in range(0, len(messages), batch_size):
            batch = messages[i : i + batch_size]
            rows = [
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "role": m["role"],
                    "content": m["content"],
                }
                for m in batch
            ]
            sb.table("chat_messages").insert(rows).execute()


def load_session_messages(sb: Client, user_id: str, session_id: str) -> list[dict]:
    result = (
        sb.table("chat_messages")
        .select("role, content")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("created_at")
        .execute()
    )
    return [{"role": r["role"], "content": r["content"]} for r in (result.data or [])]


def delete_session_from_db(sb: Client, user_id: str, session_id: str) -> None:
    sb.table("chat_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()


# ---------------------------------------------------------------------------
# Vector Store (user_id 필터 추가)
# ---------------------------------------------------------------------------
def store_vectors_batch(
    sb: Client,
    user_id: str,
    session_id: str,
    file_name: str,
    chunks: list[str],
    embeddings_list: list[list[float]],
) -> None:
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embs = embeddings_list[i : i + batch_size]
        rows = [
            {
                "user_id": user_id,
                "session_id": session_id,
                "file_name": file_name,
                "content": content,
                "metadata": {"file_name": file_name},
                "embedding": emb,
            }
            for content, emb in zip(batch_chunks, batch_embs)
        ]
        sb.table("vector_documents").insert(rows).execute()


def search_vectors(
    sb: Client, user_id: str, session_id: str, query_embedding: list[float], k: int = 5,
) -> list[Document]:
    try:
        result = sb.rpc(
            "match_vector_documents",
            {
                "query_embedding": query_embedding,
                "match_count": k,
                "filter_session_id": session_id,
                "filter_user_id": user_id,
            },
        ).execute()
        docs = []
        for row in result.data or []:
            docs.append(
                Document(
                    page_content=row["content"],
                    metadata={
                        "file_name": row["file_name"],
                        "similarity": row.get("similarity", 0),
                    },
                )
            )
        return docs
    except Exception as exc:
        logger.warning("Vector search RPC failed: %s", exc)
        result = (
            sb.table("vector_documents")
            .select("content, file_name")
            .eq("session_id", session_id)
            .eq("user_id", user_id)
            .limit(k)
            .execute()
        )
        return [
            Document(page_content=r["content"], metadata={"file_name": r["file_name"]})
            for r in (result.data or [])
        ]


def get_vector_files(sb: Client, user_id: str, session_id: str) -> list[str]:
    result = (
        sb.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .execute()
    )
    return list({row["file_name"] for row in (result.data or [])})


# ---------------------------------------------------------------------------
# PDF Processing
# ---------------------------------------------------------------------------
def process_pdf_files(
    sb: Client, user_id: str, session_id: str, uploaded_files: list[Any],
) -> list[str]:
    if not uploaded_files:
        return []
    if not OPENAI_API_KEY:
        raise ValueError("PDF 임베딩에 OPENAI_API_KEY가 필요합니다.")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", api_key=OPENAI_API_KEY,
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    processed_names: list[str] = []

    for uf in uploaded_files:
        file_name = uf.name
        suffix = Path(file_name).suffix.lower() or ".pdf"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            splits = splitter.split_documents(docs)

            if splits:
                chunks = [doc.page_content for doc in splits]
                all_embeddings: list[list[float]] = []
                for i in range(0, len(chunks), 10):
                    batch = chunks[i : i + 10]
                    all_embeddings.extend(embeddings.embed_documents(batch))

                store_vectors_batch(sb, user_id, session_id, file_name, chunks, all_embeddings)
                processed_names.append(file_name)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return processed_names


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return ChatOpenAI(
        model="gpt-4o-mini", temperature=temperature, api_key=OPENAI_API_KEY,
    )


def generate_session_title(first_q: str, first_a: str) -> str:
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.3, api_key=OPENAI_API_KEY,
        )
        prompt = (
            "다음 질문과 답변을 보고, 이 대화의 주제를 요약하는 짧은 제목(15자 이내)을 만들어주세요.\n"
            "제목만 출력하고 다른 설명은 하지 마세요.\n\n"
            f"질문: {first_q[:500]}\n답변: {first_a[:500]}"
        )
        result = llm.invoke([HumanMessage(content=prompt)])
        title = (getattr(result, "content", str(result)) or "").strip()
        return title[:50] if title else "새 세션"
    except Exception as exc:
        logger.warning("Title generation failed: %s", exc)
        return first_q[:30] if first_q else "새 세션"


def _format_memory_block(messages: list[dict], max_items: int = 50) -> str:
    tail = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자" if role == "user" else "어시스턴트"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _build_rag_messages(
    question: str, context: str, memory_text: str,
) -> list[SystemMessage | HumanMessage]:
    sys = f"""{ANSWER_STYLE_SYSTEM}

아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context}
"""
    return [SystemMessage(content=sys), HumanMessage(content=question)]


def generate_followup_questions(llm: Any, user_q: str, answer: str) -> str:
    trimmed = answer[:8000]
    prompt = (
        "다음 사용자 질문과 답변을 바탕으로, 이어서 물어볼 만한 후속 질문을 한국어로 정확히 3개만 작성하세요.\n"
        "형식:\n1. ...\n2. ...\n3. ...\n"
        "설명 문장이나 다른 텍스트는 출력하지 마세요.\n\n"
        f"[사용자 질문]\n{user_q}\n\n[답변]\n{trimmed}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = getattr(out, "content", str(out)) or ""
        raw = remove_separators(str(raw))
        if raw.strip():
            return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{raw.strip()}\n"
    except Exception as exc:
        logger.warning("Follow-up generation failed: %s", exc)
    return ""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_session() -> None:
    defaults: dict[str, Any] = {
        "current_session_id": str(uuid.uuid4()),
        "chat_history": [],
        "conversation_memory": [],
        "processed_names": [],
        "session_saved": False,
        "show_vectordb": False,
        "logged_in": False,
        "user_id": None,
        "user_login_id": None,
        "user_display_name": None,
        "auth_mode": "login",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _ensure_session_in_db(sb: Client, user_id: str, session_id: str) -> None:
    existing = (
        sb.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not existing.data:
        sb.table("chat_sessions").insert({
            "id": session_id,
            "user_id": user_id,
            "title": "새 세션 (처리 중)",
        }).execute()


# ---------------------------------------------------------------------------
# Auth UI
# ---------------------------------------------------------------------------
def _render_auth_page(sb: Client) -> bool:
    """로그인/회원가입 화면. 로그인 성공 시 True 반환."""
    st.markdown(
        """
<style>
.auth-container {
    max-width: 420px;
    margin: 80px auto 0 auto;
    padding: 2rem;
}
</style>
""",
        unsafe_allow_html=True,
    )

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=200)
        st.markdown(
            '<h2 style="text-align:center; color:#1f77b4;">재정경제부 RAG 챗봇</h2>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        tab_login, tab_register = st.tabs(["🔐 로그인", "📝 회원가입"])

        with tab_login:
            with st.form("login_form"):
                lid = st.text_input("아이디", key="login_lid")
                pwd = st.text_input("비밀번호", type="password", key="login_pwd")
                submitted = st.form_submit_button("로그인", use_container_width=True)
                if submitted:
                    if not lid or not pwd:
                        st.error("아이디와 비밀번호를 입력해주세요.")
                    else:
                        user = authenticate_user(sb, lid.strip(), pwd)
                        if user:
                            st.session_state.logged_in = True
                            st.session_state.user_id = user["id"]
                            st.session_state.user_login_id = user["login_id"]
                            st.session_state.user_display_name = user.get("display_name", lid)
                            st.success(f"환영합니다, {user.get('display_name', lid)}님!")
                            st.rerun()
                        else:
                            st.error("아이디 또는 비밀번호가 올바르지 않습니다.")

        with tab_register:
            with st.form("register_form"):
                new_lid = st.text_input("아이디", key="reg_lid")
                new_name = st.text_input("표시 이름 (선택)", key="reg_name")
                new_pwd = st.text_input("비밀번호", type="password", key="reg_pwd")
                new_pwd2 = st.text_input("비밀번호 확인", type="password", key="reg_pwd2")
                reg_submit = st.form_submit_button("회원가입", use_container_width=True)
                if reg_submit:
                    if not new_lid or not new_pwd:
                        st.error("아이디와 비밀번호를 입력해주세요.")
                    elif len(new_pwd) < 4:
                        st.error("비밀번호는 4자 이상이어야 합니다.")
                    elif new_pwd != new_pwd2:
                        st.error("비밀번호가 일치하지 않습니다.")
                    else:
                        user = register_user(sb, new_lid.strip(), new_pwd, new_name.strip())
                        if user:
                            st.success("회원가입 완료! 로그인 탭에서 로그인해주세요.")
                        else:
                            st.error("이미 존재하는 아이디입니다.")

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:  # noqa: C901
    st.set_page_config(page_title="재정경제부 RAG 챗봇", page_icon="📚", layout="wide")
    _init_session()

    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
</style>
""",
        unsafe_allow_html=True,
    )

    # ── Environment check ──
    missing_keys: list[str] = []
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    if not SUPABASE_URL:
        missing_keys.append("SUPABASE_URL")
    if not SUPABASE_ANON_KEY:
        missing_keys.append("SUPABASE_ANON_KEY")
    if missing_keys:
        st.warning(
            f"다음 환경변수가 설정되지 않았습니다: {', '.join(missing_keys)}\n\n"
            "`.env` 파일 또는 Streamlit Cloud의 Secrets를 확인해주세요."
        )
        return

    sb = get_supabase_client()
    if sb is None:
        st.error("Supabase 연결에 실패했습니다.")
        return

    # ── Auth gate ──
    if not st.session_state.logged_in:
        _render_auth_page(sb)
        return

    user_id: str = st.session_state.user_id

    # ── Header ──
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">재정경제부</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user_display_name}")
        if st.button("🚪 로그아웃", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("---")
        st.markdown("### 🤖 모델 설정")
        model_choice = st.radio("LLM 모델", ("gpt-4o-mini",), index=0)

        st.markdown("---")
        st.markdown("### 📂 세션 관리")

        sessions = load_all_sessions(sb, user_id)
        session_map: dict[str, str] = {s["id"]: s["title"] for s in sessions}

        if session_map:
            selected_session_id = st.selectbox(
                "세션 선택",
                options=list(session_map.keys()),
                format_func=lambda x: session_map.get(x, x),
                key="session_selector",
            )
        else:
            selected_session_id = None
            st.info("저장된 세션이 없습니다.")

        col_s, col_l = st.columns(2)
        with col_s:
            btn_save = st.button("💾 세션저장", use_container_width=True)
        with col_l:
            btn_load = st.button("📂 세션로드", use_container_width=True)

        col_d, col_c = st.columns(2)
        with col_d:
            btn_delete = st.button("🗑️ 세션삭제", use_container_width=True)
        with col_c:
            btn_clear = st.button("🔄 화면초기화", use_container_width=True)

        btn_vectordb = st.button("🗃️ VectorDB", use_container_width=True)

        st.markdown("---")
        st.markdown("### 📄 PDF 업로드")
        uploads = st.file_uploader(
            "PDF 파일 업로드", type=["pdf"], accept_multiple_files=True,
        )

        if st.button("📥 파일 처리하기", use_container_width=True):
            if not uploads:
                st.warning("업로드된 PDF가 없습니다.")
            else:
                with st.spinner("PDF 처리 중..."):
                    try:
                        sid = st.session_state.current_session_id
                        _ensure_session_in_db(sb, user_id, sid)
                        names = process_pdf_files(sb, user_id, sid, list(uploads))
                        st.session_state.processed_names.extend(names)
                        st.success(f"PDF 처리 완료: {', '.join(names)}")
                    except Exception as exc:
                        logger.warning("PDF 처리 실패: %s", exc)
                        st.error(f"PDF 처리 중 오류: {exc}")

        if st.session_state.processed_names:
            st.markdown("**처리된 파일**")
            for name in st.session_state.processed_names:
                st.text(f"📄 {name}")

        st.markdown("---")
        st.markdown("### ℹ️ 현재 상태")
        st.text(
            f"사용자: {st.session_state.user_display_name}\n"
            f"모델: {model_choice}\n"
            f"세션 ID: {st.session_state.current_session_id[:8]}...\n"
            f"처리된 PDF: {len(st.session_state.processed_names)}개\n"
            f"대화 기록: {len(st.session_state.conversation_memory)}개"
        )

    # ── Button handlers ──
    if btn_save:
        if not st.session_state.chat_history:
            st.warning("저장할 대화 내용이 없습니다.")
        else:
            with st.spinner("세션 저장 중..."):
                first_q, first_a = "", ""
                for m in st.session_state.chat_history:
                    if m["role"] == "user" and not first_q:
                        first_q = m["content"]
                    elif m["role"] == "assistant" and first_q and not first_a:
                        first_a = m["content"]
                        break
                title = generate_session_title(first_q, first_a)
                try:
                    save_session_to_db(
                        sb,
                        user_id,
                        st.session_state.current_session_id,
                        title,
                        st.session_state.chat_history,
                    )
                    st.session_state.session_saved = True
                    st.success(f"세션 저장 완료: '{title}'")
                    st.rerun()
                except Exception as exc:
                    st.error(f"세션 저장 실패: {exc}")

    if btn_load:
        if selected_session_id:
            with st.spinner("세션 로드 중..."):
                try:
                    messages = load_session_messages(sb, user_id, selected_session_id)
                    st.session_state.current_session_id = selected_session_id
                    st.session_state.chat_history = messages
                    st.session_state.conversation_memory = messages[-50:]
                    st.session_state.session_saved = True
                    st.session_state.processed_names = get_vector_files(
                        sb, user_id, selected_session_id,
                    )
                    st.success("세션 로드 완료!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"세션 로드 실패: {exc}")
        else:
            st.warning("로드할 세션을 선택해주세요.")

    if btn_delete:
        if selected_session_id:
            try:
                delete_session_from_db(sb, user_id, selected_session_id)
                if st.session_state.current_session_id == selected_session_id:
                    st.session_state.current_session_id = str(uuid.uuid4())
                    st.session_state.chat_history = []
                    st.session_state.conversation_memory = []
                    st.session_state.processed_names = []
                    st.session_state.session_saved = False
                st.success("세션이 삭제되었습니다.")
                st.rerun()
            except Exception as exc:
                st.error(f"세션 삭제 실패: {exc}")
        else:
            st.warning("삭제할 세션을 선택해주세요.")

    if btn_clear:
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.conversation_memory = []
        st.session_state.processed_names = []
        st.session_state.session_saved = False
        st.session_state.show_vectordb = False
        st.rerun()

    if btn_vectordb:
        st.session_state.show_vectordb = not st.session_state.show_vectordb

    # ── VectorDB info panel ──
    if st.session_state.show_vectordb:
        with st.expander("🗃️ VectorDB 파일 목록", expanded=True):
            files = get_vector_files(sb, user_id, st.session_state.current_session_id)
            if files:
                for f in files:
                    st.markdown(f"- 📄 **{f}**")
            else:
                st.info("현재 세션의 VectorDB에 저장된 파일이 없습니다.")

    # ── Chat history display ──
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]))

    # ── Chat input ──
    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_memory.append(
        {"role": "user", "content": user_input},
    )
    if len(st.session_state.conversation_memory) > 50:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]

    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    # ── Generate answer ──
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        try:
            llm = get_llm()
            has_vectors = bool(st.session_state.processed_names)

            if has_vectors:
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-ada-002", api_key=OPENAI_API_KEY,
                )
                query_emb = embeddings.embed_query(user_input)
                docs = search_vectors(
                    sb, user_id, st.session_state.current_session_id, query_emb, k=10,
                )
                context = "\n\n".join(d.page_content for d in docs)
                mem_txt = _format_memory_block(
                    st.session_state.conversation_memory[:-1],
                )
                messages = _build_rag_messages(user_input, context, mem_txt)
            else:
                mem_txt = _format_memory_block(
                    st.session_state.conversation_memory[:-1],
                )
                sys = f"{ANSWER_STYLE_SYSTEM}\n\n[대화 맥락]\n{mem_txt or '(없음)'}"
                messages = [
                    SystemMessage(content=sys),
                    HumanMessage(content=user_input),
                ]

            acc = ""
            for chunk in llm.stream(messages):
                piece = getattr(chunk, "content", "") or ""
                if piece:
                    acc += piece
                    placeholder.markdown(remove_separators(acc) + "▌")

            full_answer = remove_separators(acc)

            follow = generate_followup_questions(llm, user_input, full_answer)
            if follow:
                full_answer += follow

            placeholder.markdown(full_answer)

        except Exception as exc:
            logger.warning("답변 생성 실패: %s", exc)
            full_answer = (
                f"# 오류\n\n요청을 처리하는 중 문제가 발생했습니다.\n\n`{exc}`"
            )
            placeholder.markdown(remove_separators(full_answer))

        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_answer},
        )
        st.session_state.conversation_memory.append(
            {"role": "assistant", "content": full_answer},
        )
        if len(st.session_state.conversation_memory) > 50:
            st.session_state.conversation_memory = (
                st.session_state.conversation_memory[-50:]
            )


if __name__ == "__main__":
    main()
