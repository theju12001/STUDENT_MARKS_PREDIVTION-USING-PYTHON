ASSETS FOLDER
=============

This folder is reserved for optional static assets (e.g., a custom logo,
favicon, or additional images) that you may want to reference from the
Streamlit app or from your documentation.

No external images are required for the application to run — the premium
UI is built entirely with custom CSS and Streamlit's built-in components,
so the interface looks polished even without any files in this folder.

If you add a logo later, a common pattern is:

    st.sidebar.image("assets/logo.png", use_container_width=True)

placed near the top of the sidebar section in app.py.
