# vConDiary

## Overview

vConDiary is a Streamlit-based application for managing and processing virtual conversations (vCons). It integrates with MongoDB, CarrierX, and OpenAI for transcription and summarization.

## Prerequisites

Before setting up vConDiary, ensure you have the following installed:

- **Python 3.13.1** (Recommended, avoid system Python)
- **Virtual Environment (venv)** for dependency management
- **MongoDB** (Local or Remote)
- **OpenSSL 1.1.1+** (Check with `python3 -c "import ssl; print(ssl.OPENSSL_VERSION)"`)
- **OpenAI Python SDK** (`openai==1.3.5` recommended)
- **CarrierX API Access** (if applicable)

## Installation

### 1️⃣ Setup Virtual Environment

```sh
python3 -m venv ~/venvs/vcondiary
source ~/venvs/vcondiary/bin/activate
```

### 2️⃣ Upgrade Pip & Install Dependencies

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

### 3️⃣ Verify SSL & urllib3

```sh
python3 -c "import ssl; print(ssl.OPENSSL_VERSION)"
python3 -c "import urllib3; print(urllib3.__file__)"
```

Ensure that OpenSSL is **1.1.1+** and `urllib3` is loading from the virtual environment.

### 4️⃣ Configure Secrets

Edit `.streamlit/secrets.toml` with:

```toml
[openai]
api_key = "your-openai-key"
organization = "your-org-id"
project = "your-project-id"

[mongo_db]
url = "mongodb://localhost:27017/"
db = "vcons"
collection = "vcons"

[conserver]
api_url = "http://conserver:8000"
auth_token = "your-auth-token"
```

### 5️⃣ Run the Application

```sh
python3 -m streamlit run vcondiary.py
```

If issues persist, explicitly specify Python:

```sh
~/venvs/vcondiary/bin/python3 -m streamlit run vcondiary.py
```

## Troubleshooting

- **SSL Warning:** Ensure OpenSSL version is correct and update Python if needed.
- **Wrong Python Version:** Run `which python3` and ensure it points to `~/venvs/vcondiary/bin/python3`.
- **Streamlit Issues:** Reinstall within the virtual environment: `pip uninstall streamlit -y && pip install streamlit`.

## Contributing

Feel free to submit issues or PRs to improve vConDiary!

