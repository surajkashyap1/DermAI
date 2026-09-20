"""DermAI — dermatology chatbot with HAM10000 skin-lesion classification.

Two capabilities live under this package:

* ``dermai.classifier`` — a 7-class HAM10000 CNN with Grad-CAM explainability.
* ``dermai.rag`` — a LangChain + FAISS retrieval-augmented generation pipeline
  for grounded dermatology Q&A over curated sources.

The Streamlit entry point (``app.py`` at the repo root) wires them together into
an upload -> inference -> explanation -> chat experience.
"""

__version__ = "1.0.0"
