# vConDiary

## Overview

vConDiary is a Streamlit-based application for managing and processing virtual conversations (vCons). It integrates with MongoDB, CarrierX, and OpenAI for transcription and summarization.

## Prerequisites

Before setting up vConDiary, ensure you have the following installed:

- **Python 3.12 or later** (`vcon` 1.x requires 3.12; the Docker image and `pyproject.toml` use 3.12)
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
git clone https://github.com/vcon-dev/vcon-app-template
cd vcon-dev/vcon-app-template
```

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
streamlit run vcondiary.py
```

If running in Poetry:

```sh
poetry run streamlit run vcondiary.py
```

If issues persist, explicitly specify Python:

```sh
~/venvs/vcondiary/bin/python3 -m streamlit run vcondiary.py
```

## vCon versions

The app reads vCons with `vcon>=1.0.3,<2` (draft-ietf-vcon-vcon-core-05,
`"vcon": "0.5.0"`); release 1.0.0 moved to it from `vcon` 0.5.0. Stored vCons
that declare `"0.4.0"` are migrated to 0.5.0 by the library when they are read,
so both kinds show up in the diary. `vcon_reader.py` extracts each call's time,
parties and transcript: party entries are dictionaries, and the transcript is
taken from a `transcript` analysis (text or JSON body) or a `wtf_transcription`
analysis (a WTF document). The suggestion box reads transcripts the same way.

## Tests

```sh
pip install -r requirements.txt pytest
pytest
```

## Troubleshooting

- **SSL Warning:** Ensure OpenSSL version is correct and update Python if needed.
- **Wrong Python Version:** Run `which python3` and ensure it points to `~/venvs/vcondiary/bin/python3`.
- **Streamlit Issues:** Reinstall within the virtual environment: `pip uninstall streamlit -y && pip install streamlit`.

## Contributing

Feel free to submit issues or PRs to improve vConDiary!

