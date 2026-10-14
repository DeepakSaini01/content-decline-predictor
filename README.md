# Content decline prototype (Streamlit)

    pip install -r requirements.txt
    streamlit run app.py

Put `content_refresh_anonymized.csv` next to `app.py` (or in `data/`), or upload it in the sidebar.
Tabs: Rank content, What-if, How good is it?, Drivers.
`model_core.py` holds all modelling code (no Streamlit), so it can be reused in a notebook.
