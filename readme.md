# 🛰️ Multi-Agent Research Assistant

An AI-powered research assistant built using **LangChain**, **LangGraph**, **Mistral AI**, **Tavily Search**, **BeautifulSoup**, and **Streamlit**.

The system follows a multi-agent architecture where different agents collaborate to perform research, generate reports, and critique the final output.

---

## 🚀 Features

### 🔍 Search Agent
- Searches the web using Tavily Search API
- Retrieves recent and reliable sources
- Extracts titles, URLs, and snippets

### 📖 Reader Agent
- Selects the most relevant source
- Scrapes webpage content using BeautifulSoup
- Cleans unnecessary HTML elements
- Extracts meaningful text for analysis

### ✍️ Writer Agent
- Generates a structured research report
- Produces:
  - Introduction
  - Key Findings
  - Conclusion
  - Sources

### 🧪 Critic Agent
- Reviews the generated report
- Provides:
  - Overall score
  - Strengths
  - Areas for improvement
  - Final verdict

### 🎨 Streamlit Dashboard
- Interactive UI
- Real-time agent workflow visualization
- Search results viewer
- Scraped content viewer
- Report display
- Critic feedback section
- TXT report download

---

# 🏗️ Architecture

```text
User Topic
    │
    ▼
┌──────────────┐
│ Search Agent │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Reader Agent │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Writer Agent │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Critic Agent │
└──────────────┘
       │
       ▼
 Streamlit UI
```

---

# 📂 Project Structure

```text
Multi-Agent-Research-Assistant/
│
├── app.py
├── agents.py
├── tools.py
├── pipeline.py
├── requirements.txt
├── README.md
├── .env
│
├── __pycache__/
│
└── .venv/
```

---

# ⚙️ Tech Stack

### Frontend
- Streamlit

### AI Framework
- LangChain
- LangGraph

### LLM
- Mistral AI

### Search Engine
- Tavily Search API

### Web Scraping
- Requests
- BeautifulSoup4

### Environment Management
- Python Dotenv

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/multi-agent-research-assistant.git

cd multi-agent-research-assistant
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate virtual environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

or using uv:

```bash
uv pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
MISTRAL_API_KEY=your_mistral_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

# ▶️ Running the Application

Run Streamlit:

```bash
streamlit run app.py
```

Application opens at:

```text
http://localhost:8501
```

---

# 📝 Example Usage

Enter a topic:

```text
Impact of AI on Healthcare
```

The system will:

1. Search the web
2. Read the best source
3. Generate a detailed report
4. Critique the report

---

# 📊 Sample Output

## Research Report

```text
Introduction

Artificial Intelligence is transforming healthcare through
automation, predictive analytics, and personalized medicine.

Key Findings

1. AI improves disease detection accuracy.
2. AI reduces administrative workload.
3. AI accelerates drug discovery.

Conclusion

AI has the potential to revolutionize healthcare but
requires responsible implementation.
```

---

## Critic Feedback

```text
Score: 8/10

Strengths:
- Well structured
- Good use of sources

Areas to Improve:
- More quantitative evidence
- Include counterarguments

Verdict:
Strong report with minor improvements needed.
```

---

# 🌟 Future Improvements

- Multi-source reading instead of single URL
- Citation tracking
- RAG integration
- Report export to PDF and DOCX
- Agent memory
- Research history
- Vector database support
- Deep research mode

---

# 🤝 Contributing

Contributions are welcome.

1. Fork repository
2. Create feature branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added feature"
```

4. Push

```bash
git push origin feature-name
```

5. Create Pull Request

---


# 👩‍💻 Author

Akanksha Yadav

B.Tech CSE (AI)
IET Lucknow

Interests:
- Machine Learning
- AI Agents
- RAG Systems
- LangChain
- Generative AI
- Software Engineering
