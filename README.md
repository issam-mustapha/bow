# SmartBoW AI — Intelligent Document Search

SmartBoW AI is a document search application that combines **Bag of Words (BoW)** for text-based document retrieval with **GPT-5 nano** to generate answers based on the retrieved information.

## Features

- Search for relevant information in documents using Bag of Words.
- Process text and identify relevant documents based on word matching.
- Generate answers using the OpenAI API.
- Simple and interactive web interface built with Streamlit.

## Tech Stack

- Python
- Streamlit
- NumPy
- Pandas
- OpenAI API
- Bag of Words (BoW)

## Prerequisites

Before starting, make sure you have installed:

- Python 3.10 or later
- pip
- An OpenAI API key

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

If you already have the project on your computer, open a terminal in the project directory.

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install streamlit numpy pandas openai
```

## Configuration

Set your OpenAI API key as an environment variable.

**Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

**Linux / macOS:**

```bash
export OPENAI_API_KEY="your-api-key"
```

Replace `your-api-key` with your actual API key. Never commit your API key to GitHub or share it publicly.

## Run the Application

From the project directory, execute:

```bash
streamlit run app.py
```

Streamlit will display a local URL in your terminal, usually:

```text
http://localhost:8501
```

Open this URL in your browser to use SmartBoW AI.

## How It Works

1. **Text Processing:** The application processes the available text.
2. **Bag of Words:** Text is represented using word-frequency vectors.
3. **Document Retrieval:** The application identifies relevant documents based on the query.
4. **Answer Generation:** GPT-5 nano generates an answer using the retrieved information.
5. **Results:** The application displays the answer through the Streamlit interface.

## Project Structure

```text
SmartBoW-AI/
├── app.py
├── README.md
└── .venv/
```

The `.venv/` directory contains the local Python environment and should not be committed to Git.

## Troubleshooting

**Missing OpenAI API key**

Make sure `OPENAI_API_KEY` is configured in the terminal where you run Streamlit.

**Missing Python packages**

Activate your virtual environment and reinstall the dependencies.

**Streamlit command not found**

Run the application using:

```bash
python -m streamlit run app.py
```

**Port already in use**

Choose another port:

```bash
python -m streamlit run app.py --server.port 8502
```

## Security

- Never expose your API key in source code.
- Do not commit `.env` files containing secrets.
- Keep dependencies and API credentials secure.

## License

Add a license to this project if you intend to distribute or publish it publicly.
