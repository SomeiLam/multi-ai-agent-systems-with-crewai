# LinkedIn Content Generator

A command-line tool powered by a multi-agent AI "crew" to generate high‑engagement LinkedIn posts for any topic.

## Features

- **Multi-Agent Workflow**: Leverages three specialized agents—Planner, Writer, and Editor—for structured content planning, drafting, and polishing.
- **LinkedIn Best Practices**: Automatically formats posts with a hook, 1–2 line paragraphs, emojis, 3–5 hashtags, and a clear call-to-action.
- **CLI Interface**: Pass in your topic as a command-line argument.
- **Rich Markdown Output**: Preview formatted Markdown in your terminal using `rich` or pipe to a file.

## Requirements

- Python 3.8 or higher
- Virtual environment tool (`python -m venv`)
- An OpenAI API key

### Python Packages

Managed in `requirements.txt`:

```text
crewai==0.28.8
crewai_tools==0.1.6
langchain_community==0.0.29
openai>=0.27.0
python-dotenv>=0.21.0
rich>=12.0
```

## Setup

1. **Clone the repo**
   ```bash
   git clone <your-repo-url> linkedin_content_generator
   cd linkedin_content_generator
   ```

2. **Create & activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   # .\venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   - Copy the example template:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your OpenAI key:
     ```text
     OPENAI_API_KEY=your_openai_api_key_here
     OPENAI_MODEL_NAME=gpt-3.5-turbo
     ```

## Usage

Generate a LinkedIn post by passing your topic:

```bash
python main.py "How AI is transforming cybersecurity"
```
