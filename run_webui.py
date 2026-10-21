"""
Office Agent Web 界面启动入口

启动 Gradio 聊天界面，默认地址：http://127.0.0.1:7860
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.startup import check_environment
from src.webui.app import demo, CUSTOM_THEME, CUSTOM_CSS


def main():
    check_environment()

    print("=" * 50)
    print("  Office Agent Web 界面")
    print("  启动地址: http://127.0.0.1:7860")
    print("=" * 50)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=CUSTOM_THEME,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
