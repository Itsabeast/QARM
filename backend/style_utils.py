import streamlit as st

def apply_sidebar_style():
    """
    Applies professional, theme-aware CSS to the Streamlit Sidebar.
    Works in both Light Mode and Dark Mode.
    """
    st.markdown("""
        <style>
        /* --- SIDEBAR STYLING --- */
        
        /* Sidebar Background (Transparent Grey - works in Dark/Light) */
        [data-testid="stSidebar"] {
            background-color: rgba(128, 128, 128, 0.05);
            border-right: 1px solid rgba(128, 128, 128, 0.1);
        }

        /* Navigation Section Padding */
        [data-testid="stSidebarNav"] {
            padding-top: 1rem;
        }

        /* Link Buttons */
        [data-testid="stSidebarNav"] a {
            background-color: transparent;
            border-radius: 8px;
            margin-bottom: 8px;
            padding: 12px 15px;
            transition: all 0.2s ease-in-out;
            border: 1px solid transparent;
            color: inherit; /* Important: Inherits white in dark mode, black in light mode */
        }

        /* Hover Effect */
        [data-testid="stSidebarNav"] a:hover {
            background-color: rgba(128, 128, 128, 0.1); /* Subtle hover tint */
            border: 1px solid rgba(128, 128, 128, 0.2);
            transform: translateX(5px);
        }

        /* Active Page Highlight */
        [data-testid="stSidebarNav"] [aria-current="page"] {
            background-color: rgba(59, 130, 246, 0.15); /* Transparent Blue */
            border: 1px solid #3b82f6; /* Bright Blue Border */
            font-weight: 700;
        }

        /* Text Styling */
        [data-testid="stSidebarNav"] span {
            font-size: 1.1rem !important;
            font-weight: 500;
            color: inherit; /* Inherits correct theme color */
        }

        /* Logo Area Spacing */
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
