import webview
import subprocess
import sys
import os
import time
import socket


def is_port_in_use(port=8501):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def main():
    app_path = resource_path("app.py")

    process = None

    if not is_port_in_use():
        process = subprocess.Popen([
            sys.executable,
            "-m", "streamlit",
            "run", app_path,
            "--server.headless=true",
            "--server.runOnSave=false",
            "--server.fileWatcherType=none"
        ])

    time.sleep(3)

    webview.create_window(
        "🧠 Brain Tumor AI",
        "http://127.0.0.1:8501",
        width=1200,
        height=800
    )

    webview.start()

    if process:
        process.terminate()


if __name__ == "__main__":
    main()