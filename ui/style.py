import streamlit as st

def inject_global_css() -> None:
    st.markdown(
        """
        <style>
          /* Hide Streamlit's default pages list */
          [data-testid="stSidebarNav"] { display: none; }

          /* Optional: reduce extra spacing that stSidebarNav adds */
          [data-testid="stSidebarNav"] + div { margin-top: 0rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )