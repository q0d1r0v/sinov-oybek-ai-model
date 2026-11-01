# Sinov-Oybek

An experimental Uzbek language model by Digital Group. It has 10 million parameters, was trained from scratch, and runs locally on Linux with no internet connection.

> **This is a test model, not a finished product.** It answers in Uzbek and its answers are often wrong: 5 of 36 knowledge checks and 0 of 15 math checks were correct. It also tends to introduce itself when asked "who is X?" questions. Verify its answers before using them.

## Requirements

**Linux only for now.** Windows and macOS are not supported yet.

- Linux with Python 3.10+ and `python3-venv`
- 4 GB RAM (8 GB recommended) and a dual-core CPU
- No GPU required
- About 1.2 GB of free disk space and internet access for installation

## Install and run

```bash
git clone https://github.com/q0d1r0v/sinov-oybek-ai-model.git
cd sinov-oybek-ai-model
./install.sh
./run.sh
```

Then open <http://127.0.0.1:8000> in your browser.

To chat in the terminal instead, run `.venv/bin/python chat.py`.

## Uncensored model

This model is uncensored: it was not taught to refuse questions, and the app has no content filter. Only 1 of 150 answers in the 50-question test contained a refusal marker.

- The model can give offensive, false, or one-sided answers.
- Views found in its training texts can appear in its answers.
- You are responsible for how you use its answers. Do not give it to children.

## Privacy

The server listens on `127.0.0.1` only, so other computers cannot reach it. Conversations are not stored or sent anywhere.
