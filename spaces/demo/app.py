"""Supplemental hosted research demo; models load only in a bounded request worker."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
W4A8_LIBRARY = ROOT / "build" / "w4a8" / "libitantra_w4a8.so"

if W4A8_LIBRARY.is_file() and (ROOT / "tools" / "unified_research_web.py").is_file():
    from tools.unified_research_web import DEMO_CSS, build_demo

    demo = build_demo(W4A8_LIBRARY, hosted=True)
else:
    from itantra_tts.web.demo import create_default_demo

    demo = create_default_demo(hosted=True)
    DEMO_CSS = None

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), show_error=False,
        css=DEMO_CSS,
    )
