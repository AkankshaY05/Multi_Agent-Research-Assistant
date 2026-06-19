"""
Multi-Agent Research Assistant — Streamlit UI
================================================
Front-end for a 4-agent pipeline: Search -> Reader -> Writer -> Critic.

IMPORTANT — read before relying on this in production:
The underlying agents (agents.py) return free-form text, not structured
data. Search results, the "selected source URL", and the critic's score
are extracted from that text with regex heuristics below. They will
degrade gracefully (show "Not detected") rather than crash if parsing
fails, but they are NOT guaranteed accurate. For reliable structured
output, change the search/critic agent prompts to return strict JSON.
"""

import io
import re
import time

import streamlit as st
import streamlit.components.v1 as components

from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain


# ----------------------------------------------------------------------------
# Page config & session state
# ----------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "state": None,          # final pipeline outputs
        "agent_status": {       # per-agent status + timing
            "search": {"status": "pending", "seconds": None},
            "reader": {"status": "pending", "seconds": None},
            "writer": {"status": "pending", "seconds": None},
            "critic": {"status": "pending", "seconds": None},
        },
        "error": None,
        "running": False,
        "theme": "dark",
        "total_time": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------

def inject_css(theme: str):
    if theme == "dark":
        bg = "#0a0e1a"
        bg_grad = "linear-gradient(135deg, #0a0e1a 0%, #131a2e 100%)"
        card_bg = "rgba(255,255,255,0.04)"
        border = "rgba(0,255,255,0.18)"
        text = "#e6f1ff"
        subtext = "#8fa3c4"
        accent = "#00e5ff"
        accent2 = "#a855f7"
    else:
        bg = "#f4f7fb"
        bg_grad = "linear-gradient(135deg, #f4f7fb 0%, #e9eef7 100%)"
        card_bg = "rgba(255,255,255,0.85)"
        border = "rgba(0,120,200,0.18)"
        text = "#0f172a"
        subtext = "#475569"
        accent = "#0284c7"
        accent2 = "#7c3aed"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background: {bg_grad};
            color: {text};
        }}

        h1, h2, h3, .hero-title {{
            font-family: 'Space Grotesk', sans-serif;
        }}

        .hero-title {{
            font-size: 2.6rem;
            font-weight: 700;
            background: linear-gradient(90deg, {accent}, {accent2});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }}

        .hero-sub {{
            color: {subtext};
            font-size: 1.05rem;
            margin-top: 0.2rem;
        }}

        .glass-card {{
            background: {card_bg};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.9rem;
            backdrop-filter: blur(6px);
        }}

        .agent-step {{
            text-align: center;
            padding: 0.9rem 0.4rem;
            border-radius: 12px;
            border: 1px solid {border};
            background: {card_bg};
        }}

        .status-pending {{ color: {subtext}; }}
        .status-running {{ color: {accent}; font-weight: 600; }}
        .status-completed {{ color: #22c55e; font-weight: 600; }}
        .status-error {{ color: #ef4444; font-weight: 600; }}

        .score-badge {{
            display: inline-block;
            font-size: 1.8rem;
            font-weight: 700;
            color: {accent};
            border: 2px solid {accent};
            border-radius: 50%;
            width: 80px;
            height: 80px;
            line-height: 76px;
            text-align: center;
        }}

        .scroll-box {{
            max-height: 320px;
            overflow-y: auto;
            background: {card_bg};
            border: 1px solid {border};
            border-radius: 10px;
            padding: 1rem;
            font-size: 0.92rem;
            color: {text};
        }}

        .url-highlight {{
            color: {accent};
            font-weight: 600;
            word-break: break-all;
        }}

        .stat-number {{
            font-size: 1.7rem;
            font-weight: 700;
            color: {accent};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🛰️ Project Info")
        st.markdown(
            "A 4-agent research pipeline:\n\n"
            "1. **Search Agent** — finds recent, reliable sources\n"
            "2. **Reader Agent** — scrapes the best source\n"
            "3. **Writer Agent** — drafts a structured report\n"
            "4. **Critic Agent** — scores and critiques the report"
        )
        st.divider()
        st.markdown("### ⚙️ Settings")
        theme = st.radio("Theme", ["dark", "light"], index=0, horizontal=True)
        st.caption(
            "Note: search-result cards and the critic score are parsed "
            "heuristically from free-text agent output — treat as best-effort."
        )
        st.divider()
        st.caption("Built with Streamlit · LangChain agents")
    return theme


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------

def render_header():
    st.markdown('<div class="hero-title">Multi-Agent Research Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Search → Read → Write → Critique — '
        'an autonomous agent pipeline that turns a topic into a vetted report.</div>',
        unsafe_allow_html=True,
    )
    st.write("")


# ----------------------------------------------------------------------------
# Input section
# ----------------------------------------------------------------------------

def render_input_section():
    col1, col2 = st.columns([5, 1])
    with col1:
        topic = st.text_input(
            "Research topic",
            placeholder="Impact of AI on Healthcare",
            label_visibility="collapsed",
        )
    with col2:
        run_clicked = st.button(
            "🚀 Research",
            use_container_width=True,
            disabled=st.session_state.running,
        )
    return topic, run_clicked


# ----------------------------------------------------------------------------
# Agent progress section
# ----------------------------------------------------------------------------

AGENT_LABELS = {
    "search": "🔍 Search Agent",
    "reader": "📖 Reader Agent",
    "writer": "✍️ Writer Agent",
    "critic": "🧪 Critic Agent",
}

ICONS = {"pending": "⏳", "running": "⚙️", "completed": "✅", "error": "❌"}


def render_agent_progress(placeholder):
    """Render the 4-step horizontal workflow from current session state."""
    with placeholder.container():
        cols = st.columns(4)
        for col, key in zip(cols, ["search", "reader", "writer", "critic"]):
            info = st.session_state.agent_status[key]
            status = info["status"]
            seconds = info["seconds"]
            time_str = f"{seconds:.1f}s" if seconds is not None else "—"
            col.markdown(
                f"""
                <div class="agent-step">
                    <div style="font-size:1.4rem;">{ICONS[status]}</div>
                    <div style="font-weight:600;">{AGENT_LABELS[key]}</div>
                    <div class="status-{status}">{status.capitalize()}</div>
                    <div style="font-size:0.8rem; opacity:0.7;">{time_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------------
# Pipeline execution with live progress
# ----------------------------------------------------------------------------

def reset_agent_status():
    for key in st.session_state.agent_status:
        st.session_state.agent_status[key] = {"status": "pending", "seconds": None}


def execute_pipeline(topic: str, progress_placeholder):
    """Runs the 4 agents sequentially, updating status/timing as it goes."""
    reset_agent_status()
    result = {}
    pipeline_start = time.time()

    try:
        # Step 1: Search
        st.session_state.agent_status["search"]["status"] = "running"
        render_agent_progress(progress_placeholder)
        t0 = time.time()
        search_agent = build_search_agent()
        search_result = search_agent.invoke(
            {"messages": [("user", f"Find recent, reliable and detailed information about {topic}")]}
        )
        result["search_result"] = extract_text(search_result["messages"][-1].content)
        st.session_state.agent_status["search"] = {"status": "completed", "seconds": time.time() - t0}
        render_agent_progress(progress_placeholder)

        # Step 2: Reader
        st.session_state.agent_status["reader"]["status"] = "running"
        render_agent_progress(progress_placeholder)
        t0 = time.time()
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search results about '{topic}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{result['search_result'][:800]}",
                    )
                ]
            }
        )
        result["scraped_content"] = extract_text(reader_result["messages"][-1].content)
        st.session_state.agent_status["reader"] = {"status": "completed", "seconds": time.time() - t0}
        render_agent_progress(progress_placeholder)

        # Step 3: Writer
        st.session_state.agent_status["writer"]["status"] = "running"
        render_agent_progress(progress_placeholder)
        t0 = time.time()
        research_combined = (
            f"SEARCH RESULTS:\n{result['search_result']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{result['scraped_content']}"
        )
        report_raw = writer_chain.invoke({"topic": topic, "research": research_combined})
        result["report"] = extract_text(report_raw)
        st.session_state.agent_status["writer"] = {"status": "completed", "seconds": time.time() - t0}
        render_agent_progress(progress_placeholder)

        # Step 4: Critic
        st.session_state.agent_status["critic"]["status"] = "running"
        render_agent_progress(progress_placeholder)
        t0 = time.time()
        feedback_raw = critic_chain.invoke({"report": result["report"]})
        result["feedback"] = extract_text(feedback_raw)
        st.session_state.agent_status["critic"] = {"status": "completed", "seconds": time.time() - t0}
        render_agent_progress(progress_placeholder)

        st.session_state.total_time = time.time() - pipeline_start
        st.session_state.state = result
        st.session_state.error = None

    except Exception as e:
        # Mark whichever agent was running as errored
        for key, info in st.session_state.agent_status.items():
            if info["status"] == "running":
                info["status"] = "error"
        render_agent_progress(progress_placeholder)
        st.session_state.error = str(e)
        st.session_state.state = None


def extract_text(value):
    """Normalize LangChain message objects/lists or plain strings to text.

    Some agent setups return .content as a list of content blocks
    (e.g. [{"type": "text", "text": "..."}]) instead of a plain string.
    """
    if isinstance(value, str):
        return value
    if hasattr(value, "content"):
        value = value.content
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)


# ----------------------------------------------------------------------------
# Heuristic parsers (best-effort — see module docstring)
# ----------------------------------------------------------------------------

URL_PATTERN = re.compile(r"https?://[^\s\)\]\"]+")


def parse_search_results(raw_text: str):
    """Best-effort extraction of (url, snippet) pairs from free-text search output."""
    if not raw_text:
        return []
    urls = URL_PATTERN.findall(raw_text)
    # Split text into rough chunks around each URL occurrence for a snippet.
    results = []
    seen = set()
    for url in urls:
        clean_url = url.rstrip(".,);")
        if clean_url in seen:
            continue
        seen.add(clean_url)
        idx = raw_text.find(url)
        snippet = raw_text[max(0, idx - 120): idx + 120].replace("\n", " ").strip()
        title_guess = clean_url.split("//")[-1].split("/")[0]
        results.append({"title": title_guess, "url": clean_url, "snippet": snippet})
    return results


def parse_critic_feedback(raw_text: str):
    """Best-effort extraction of score / strengths / improvements from critic text."""
    parsed = {"score": None, "strengths": [], "improvements": []}
    if not raw_text:
        return parsed

    score_match = re.search(r"(\d{1,2})\s*/\s*10", raw_text)
    if score_match:
        parsed["score"] = int(score_match.group(1))

    strengths_match = re.search(
        r"strengths?[:\-]?\s*(.*?)(?:improvements?|weaknesses?|suggestions?|$)",
        raw_text, re.IGNORECASE | re.DOTALL,
    )
    improvements_match = re.search(
        r"(?:improvements?|weaknesses?|suggestions?)[:\-]?\s*(.*)",
        raw_text, re.IGNORECASE | re.DOTALL,
    )

    def to_bullets(block):
        if not block:
            return []
        lines = re.split(r"\n|•|-\s", block)
        return [l.strip(" -•\n") for l in lines if l.strip(" -•\n")][:6]

    parsed["strengths"] = to_bullets(strengths_match.group(1)) if strengths_match else []
    parsed["improvements"] = to_bullets(improvements_match.group(1)) if improvements_match else []
    return parsed


# ----------------------------------------------------------------------------
# Result rendering
# ----------------------------------------------------------------------------

def render_search_results(raw_text: str):
    results = parse_search_results(raw_text)
    with st.expander(f"🔍 Search Results ({len(results)} sources detected)", expanded=False):
        if not results:
            st.info("No URLs detected in the search agent's output.")
        for r in results:
            st.markdown(
                f"""
                <div class="glass-card">
                    <b>{r['title']}</b><br>
                    <span class="url-highlight">{r['url']}</span><br>
                    <span style="opacity:0.8;">{r['snippet']}…</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    return results


def render_scraped_content(raw_text: str, selected_url: str):
    with st.expander("📖 Scraped Content", expanded=False):
        if selected_url:
            st.markdown(f'**Selected source:** <span class="url-highlight">{selected_url}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="scroll-box">{raw_text}</div>', unsafe_allow_html=True)


def render_report(report_text: str):
    st.markdown("### 📄 Research Report")
    tabs = st.tabs(["Formatted", "Raw / Copy"])
    with tabs[0]:
        st.markdown(f'<div class="glass-card">{report_text}</div>', unsafe_allow_html=True)
    with tabs[1]:
        st.text_area("Raw report text", report_text, height=240, label_visibility="collapsed")
        copy_to_clipboard_button(report_text)

    st.download_button(
        "⬇️ Download as TXT",
        data=report_text,
        file_name="research_report.txt",
        mime="text/plain",
        use_container_width=True,
    )


def copy_to_clipboard_button(text: str):
    safe_text = text.replace("\\", "\\\\").replace("`", "\\`").replace("</script>", "<\\/script>")
    components.html(
        f"""
        <button onclick="navigator.clipboard.writeText(`{safe_text}`)"
            style="padding:0.5rem 1rem; border-radius:8px; border:1px solid #00e5ff;
                   background:transparent; color:#00e5ff; cursor:pointer;">
            📋 Copy report to clipboard
        </button>
        """,
        height=50,
    )



def render_critic_feedback(raw_text: str):
    parsed = parse_critic_feedback(raw_text)
    st.markdown("### 🧪 Critic Feedback")
    col1, col2 = st.columns([1, 3])
    with col1:
        score_display = parsed["score"] if parsed["score"] is not None else "N/A"
        st.markdown(f'<div class="score-badge">{score_display}</div>', unsafe_allow_html=True)
        st.caption("Score (parsed heuristically)")
    with col2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card"><b>✅ Strengths</b></div>', unsafe_allow_html=True)
            if parsed["strengths"]:
                for s in parsed["strengths"]:
                    st.markdown(f"- {s}")
            else:
                st.caption("None detected — see raw feedback below.")
        with c2:
            st.markdown('<div class="glass-card"><b>🛠 Improvements</b></div>', unsafe_allow_html=True)
            if parsed["improvements"]:
                for i in parsed["improvements"]:
                    st.markdown(f"- {i}")
            else:
                st.caption("None detected — see raw feedback below.")
    with st.expander("Raw critic output"):
        st.write(raw_text)


def render_stats(state: dict, search_results: list):
    st.markdown("### 📊 Statistics")
    report_text = state.get("report", "") or ""
    word_count = len(report_text.split())
    selected_url_match = URL_PATTERN.search(state.get("scraped_content", "") or "")
    selected_url = selected_url_match.group(0) if selected_url_match else "Not detected"
    total_time = st.session_state.total_time

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="glass-card"><div class="stat-number">{len(search_results)}</div>Sources found</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="glass-card"><div class="stat-number">{word_count}</div>Report word count</div>', unsafe_allow_html=True)
    with c3:
        time_str = f"{total_time:.1f}s" if total_time else "—"
        st.markdown(f'<div class="glass-card"><div class="stat-number">{time_str}</div>Total processing time</div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="glass-card" style="word-break:break-all; font-size:0.8rem;"><b>Selected source</b><br>{selected_url}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🛰️", layout="wide")
    init_session_state()

    theme = render_sidebar()
    st.session_state.theme = theme
    inject_css(theme)

    render_header()
    topic, run_clicked = render_input_section()

    progress_placeholder = st.empty()
    render_agent_progress(progress_placeholder)  # initial pending state

    if run_clicked:
        if not topic.strip():
            st.warning("Enter a research topic first.")
        else:
            st.session_state.running = True
            execute_pipeline(topic, progress_placeholder)
            st.session_state.running = False

    if st.session_state.error:
        st.error(f"Pipeline failed: {st.session_state.error}")

    state = st.session_state.state
    if state:
        st.divider()
        search_results = render_search_results(state.get("search_result", ""))
        selected_url_match = URL_PATTERN.search(state.get("scraped_content", "") or "")
        render_scraped_content(state.get("scraped_content", ""), selected_url_match.group(0) if selected_url_match else None)
        st.divider()
        render_report(state.get("report", ""))
        st.divider()
        render_critic_feedback(state.get("feedback", ""))
        st.divider()
        render_stats(state, search_results)
    elif not st.session_state.error:
        st.info("Enter a topic above and click **Research** to start the pipeline.")


if __name__ == "__main__":
    main()