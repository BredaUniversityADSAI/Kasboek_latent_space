from flask import Flask, render_template, request
import subprocess
import threading
import time

app = Flask(__name__)

latest_file = None
auto_print_timer = None
TIMEOUT_SECONDS = 30  # change this to whatever you want


def auto_print_task(filename):
    time.sleep(TIMEOUT_SECONDS)

    # If user STILL hasn't cancelled or printed
    if latest_file == filename:
        print("Timeout reached — auto printing.")
        subprocess.run(["lp", filename])


@app.route("/")
def home():
    return render_template("index.html", file=latest_file)


@app.route("/notify", methods=["POST"])
def notify():
    global latest_file, auto_print_timer

    latest_file = request.json["filename"]

    # Start a new timer
    if auto_print_timer:
        auto_print_timer.cancel()

    auto_print_timer = threading.Timer(TIMEOUT_SECONDS, auto_print_task, args=[latest_file])
    auto_print_timer.start()

    return {"status": "updated"}


@app.route("/print", methods=["POST"])
def print_file():
    global latest_file, auto_print_timer

    if auto_print_timer:
        auto_print_timer.cancel()

    if latest_file:
        subprocess.run(["lp", latest_file])

    latest_file = None
    return {"status": "printed"}


@app.route("/cancel", methods=["POST"])
def cancel():
    global latest_file, auto_print_timer

    if auto_print_timer:
        auto_print_timer.cancel()

    latest_file = None
    return {"status": "cancelled"}


app.run(host="0.0.0.0", port=5000)
