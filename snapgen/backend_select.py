
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in project root

QAI_HUB_API_TOKEN = os.getenv("QAI_HUB_API_TOKEN")


def get_execution_providers(backend: str = "qnn"):
    if backend == "qnn":
        return [("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"})]
    elif backend == "directml":
        return ["DmlExecutionProvider"]
    return ["CPUExecutionProvider"]


def qai_hub_login():
    """Only needed if compiling/profiling on Qualcomm's cloud device farm."""
    import qai_hub as hub
    if not QAI_HUB_API_TOKEN:
        raise ValueError("Set QAI_HUB_API_TOKEN in your .env file")
    hub.set_api_token(QAI_HUB_API_TOKEN)