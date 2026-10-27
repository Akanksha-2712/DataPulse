"""Main entrypoint for the Data Quality Monitoring & Anomaly Detection System.

This application is built with Streamlit to provide an interactive dashboard
for data quality profiling, Pandera schema validation, distribution drift
monitoring (Evidently / SciPy), and unsupervised anomaly detection (Isolation Forest).

Stage 1: Project scaffolding and dependency setup.
Application features will be connected in subsequent stages.
"""

import sys
from pathlib import Path

# Add project root and src to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Entry point placeholder for Streamlit application."""
    print("Data Quality Monitoring and Anomaly Detection System")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python interpreter: {sys.executable}")
    print("Application scaffolding initialized successfully.")


if __name__ == "__main__":
    main()
