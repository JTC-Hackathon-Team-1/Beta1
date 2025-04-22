Here is your setup.md optimized for your M4 Mac with 48GB RAM:

⸻

🛠️ CasaLingua Setup Guide (Optimized for M4 Mac, 48GB RAM)

This guide walks you through installing and running CasaLingua v21 on your Apple Silicon M4 MacBook Pro.

⸻

🚀 1. System Requirements
	•	macOS Sonoma or later
	•	Python 3.10+ (via Pyenv or Homebrew)
	•	48GB RAM (enables large LLMs like BLOOMZ)
	•	Apple Silicon (M4) — Accelerated with transformers + torch

⸻

🐍 2. Python Environment Setup

Option A: Using venv

python3.10 -m venv casalingua_env
source casalingua_env/bin/activate
pip install --upgrade pip setuptools wheel

Option B: Using pyenv

brew install pyenv
pyenv install 3.10.13
pyenv virtualenv 3.10.13 casalingua_env
pyenv activate casalingua_env



⸻

📦 3. Install Requirements

cd casalingua
pip install -r requirements.txt

For Apple Silicon:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers



⸻

🌍 4. Run Admin Panel (Web UI)

cd admin_panel
uvicorn main:app --reload --port 8000

Visit: http://localhost:8000

⸻

📚 5. Download Optional Hugging Face Models

python3
>>> from transformers import pipeline
>>> pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> AutoTokenizer.from_pretrained("bigscience/bloomz-1b7")
>>> AutoModelForCausalLM.from_pretrained("bigscience/bloomz-1b7")



⸻

🧪 6. Run a Test Translation

python main.py

You should see output with translated text and governance audit.

⸻

🧼 7. Optional Cleanup

find . -name '*.pyc' -delete
rm -rf __pycache__ .DS_Store



⸻

✅ You’re now ready to run CasaLingua locally with full LLM support and a dynamic admin panel.
