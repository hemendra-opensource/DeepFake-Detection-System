"""
dashboard/components/loader.py
================================
AI SaaS styled loading spinner for DeepFake detection status events.
"""

import streamlit as st


def render_ai_loader(status: str) -> None:
    """
    Render a premium, animated AI loading spinner with the current task state.

    Args:
        status: The message to show (e.g. 'Analyzing Face...', 'Running XceptionNet...')
    """
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 2.5rem 1.5rem;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            backdrop-filter: blur(12px);
            margin: 1.5rem 0;
            animation: fadeIn 0.4s ease-out;
        ">
            <div class="loader-ring"></div>
            <div style="
                font-family: var(--font-heading);
                font-weight: 700;
                font-size: 1.35rem;
                color: var(--primary-light);
                margin-top: 1rem;
                letter-spacing: -0.01em;
            ">
                🤖 DETECTING...
            </div>
            <div style="
                color: var(--text-secondary);
                font-size: 0.95rem;
                margin-top: 0.5rem;
                font-weight: 500;
            ">
                {status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
