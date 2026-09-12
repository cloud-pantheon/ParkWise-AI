# 🌲 ParkWise AI — Agentic RAG National Park Assistant

ParkWise AI is an intelligent **Agentic Retrieval-Augmented Generation (Agentic RAG)** application designed to answer questions about U.S. National Parks using trusted park documents.

Unlike a traditional RAG system that performs a single retrieval step, ParkWise AI uses an AI agent to **understand the user's intent, plan a search strategy, retrieve relevant information, evaluate the retrieved evidence, reformulate the query when necessary, and generate a grounded response**.

The application is built with **Python, Streamlit, Gemini, Sentence Transformers, and vector-based semantic retrieval**.

---

## 🚀 Live Demo

parkwise.up.railway.app

```text
https://your-app-link.streamlit.app
```

---

## 🧠 What Makes ParkWise Agentic?

Traditional RAG normally follows:

```text
User Question
      ↓
Embedding
      ↓
Vector Search
      ↓
Top Documents
      ↓
LLM
      ↓
Answer
```

ParkWise AI uses an Agentic RAG workflow:

```text
User Question
      ↓
Planning Agent
      ↓
Intent Detection
      ↓
Query Rewriting
      ↓
Document / Park Selection
      ↓
Semantic Retrieval
      ↓
Evidence Evaluation
      ↓
Is the evidence sufficient?
     ↙                 ↘
   Yes                  No
    ↓                    ↓
Generate Answer      Rewrite Query
                         ↓
                    Retrieve Again
                         ↓
                    Re-evaluate
                         ↓
                    Generate Answer
```

This allows ParkWise AI to handle more complex, vague, multi-part, and multi-document questions.

---

## ✨ Key Features

* 🤖 Agentic RAG architecture
* 🔍 Semantic document retrieval
* 🧠 AI-powered query planning
* ✏️ Automatic query rewriting
* 📚 Multi-document retrieval
* 🌲 Multi-park comparison
* 🔁 Automatic retrieval retry
* ✅ Evidence quality evaluation
* 🧩 Multi-step reasoning
* 📖 Grounded answers based on park documents
* 🧠 Agent Decision Trace
* ⚡ Cached resources for improved performance
* 💬 Interactive Streamlit interface
* 🔐 Secure API key management using environment variables

---

## 🔄 Traditional RAG vs Agentic RAG

| Feature               | Traditional RAG  | ParkWise Agentic RAG        |
| --------------------- | ---------------- | --------------------------- |
| Query understanding   | Basic            | Advanced                    |
| Retrieval             | Single retrieval | Multiple retrieval attempts |
| Query rewriting       | ❌                | ✅                           |
| Intent detection      | Limited          | ✅                           |
| Multi-document search | Limited          | ✅                           |
| Evidence evaluation   | ❌                | ✅                           |
| Retry retrieval       | ❌                | ✅                           |
| Planning              | ❌                | ✅                           |
| Dynamic workflow      | ❌                | ✅                           |
| Multi-step reasoning  | Limited          | ✅                           |
| Agent trace           | ❌                | ✅                           |

---

## 🛠️ Tech Stack

### Programming Language

* Python

### AI / Machine Learning

* Google Gemini
* Sentence Transformers
* Semantic Search
* Embeddings
* Retrieval-Augmented Generation
* Agentic RAG

### Frontend

* Streamlit

### Data Processing

* PDF document processing
* Text chunking
* Vector embeddings
* Similarity search

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │      User Query     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Planning Agent    │
                    │                     │
                    │ • Detect intent     │
                    │ • Select park       │
                    │ • Rewrite query     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Retrieval Engine    │
                    │                     │
                    │ • Embeddings        │
                    │ • Semantic search   │
                    │ • Keyword matching  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Evidence Evaluator  │
                    └──────────┬──────────┘
                               │
                   ┌───────────┴───────────┐
                   │                       │
             Sufficient                Insufficient
                   │                       │
                   ▼                       ▼
        ┌──────────────────┐     ┌──────────────────┐
        │ Generate Answer  │     │ Rewrite / Replan │
        └────────┬─────────┘     └────────┬─────────┘
                 │                        │
                 │                        ▼
                 │                Retrieve Again
                 │                        │
                 └────────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │    Final Answer     │
                    └─────────────────────┘
```

---

## 🧠 Agent Workflow

ParkWise AI follows a multi-stage reasoning workflow.

### 1. Planning

The AI analyzes the user's question and determines:

* User intent
* Relevant national park
* Important concepts
* Search strategy
* Rewritten retrieval query

### 2. Retrieval

The system converts the search query into an embedding and retrieves the most relevant chunks from the park documents.

### 3. Evidence Evaluation

The agent evaluates whether the retrieved information is sufficient to answer the question accurately.

### 4. Replanning

If the retrieved evidence is insufficient, the agent modifies the search query and performs another retrieval.

### 5. Evidence Accumulation

Useful evidence from multiple retrieval attempts can be combined instead of discarding previous results.

### 6. Response Generation

Gemini generates the final answer using the retrieved evidence.

---

## 💡 Example Questions

### Simple Retrieval

```text
Can I bring my dog to Redwood National Park?
```

```text
What are the camping rules at Mount Rainier?
```

### Complex Questions

```text
Are dogs allowed on hiking trails in Redwood National Park?
```

```text
What should a first-time visitor know before hiking at Mount Rainier?
```

### Multi-Part Questions

```text
Can I bring my dog and camp overnight at Redwood National Park?
```

### Multi-Park Comparison

```text
Compare the hiking rules of Redwood National Park and Rocky Mountain National Park.
```

### Agentic Reasoning Test

```text
I want to travel with my dog, go hiking, and camp overnight.
Between Redwood National Park and Rocky Mountain National Park,
which park would be more suitable for me and why?
```

This type of question demonstrates the benefit of Agentic RAG because the system may need to retrieve information about:

```text
Pet Rules
   +
Hiking Rules
   +
Camping Rules
   +
Multiple Parks
   ↓
Comparison
   ↓
Recommendation
```

---

## 🧠 Agent Decision Trace

ParkWise AI includes an optional **Agent Decision Trace** in the Streamlit interface.

It can display information such as:

```text
PLAN

Intent:
Compare parks for pet-friendly hiking and camping.

Target Documents:
Redwood
Rocky Mountain

Search Query:
pet rules hiking camping overnight restrictions
```

Followed by:

```text
RETRIEVE
   ↓
EVALUATE
   ↓
REPLAN
   ↓
RETRIEVE AGAIN
   ↓
GENERATE ANSWER
```

This feature is especially useful for demonstrating how Agentic RAG differs from traditional RAG.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ParkWise-AI.git
cd ParkWise-AI
```

---

### 2. Create a virtual environment

#### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file inside the project directory.

```env
GEMINI_API_KEY=your_gemini_api_key
```

Do not upload your `.env` file to GitHub.

Add it to `.gitignore`:

```text
.env
.venv/
__pycache__/
```

---

## ▶️ Run the Application

Run:

```bash
python -m streamlit run app.py
```

The application should open at:

```text
http://localhost:8501
```

---

## 📂 Project Structure

A typical project structure looks like:

```text
ParkWise-AI/
│
├── app.py
│
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
├── data/
│   └── National Park PDF documents
│
├── src/
│   ├── pdf_loader.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── agent.py
│   └── generator.py
│
└── vector / embedding data
```

The exact structure may vary as the project evolves.

---

## 🔍 Semantic Retrieval

ParkWise converts document chunks into vector embeddings.

When a user asks a question:

```text
Question
   ↓
Sentence Transformer
   ↓
Query Embedding
   ↓
Similarity Search
   ↓
Relevant Document Chunks
```

This allows ParkWise to find semantically related information even when the wording of the user's question is different from the wording inside the source documents.

---

## 🔁 Agentic Retrieval

One of the major improvements over basic RAG is automatic retrieval retry.

For example:

```text
User:
Can I take my dog hiking in Redwood?
```

The first search may be:

```text
dog hiking Redwood
```

If the evidence is incomplete, the agent may automatically rewrite it as:

```text
pets dogs leash allowed prohibited hiking trails Redwood National Park
```

The system then retrieves additional evidence before generating the final response.

---

## ⚡ Performance Optimization

ParkWise can cache expensive resources such as:

* Embedding models
* Document data
* Vector indexes
* Processed document chunks

This prevents the system from unnecessarily recreating these resources for every user request.

Example:

```python
@st.cache_resource
def load_embedding_model():
    ...
```

Caching significantly improves response time after the application has started.

---

## 🌐 Deployment

ParkWise can be deployed using platforms such as:

* Streamlit Community Cloud
* Railway
* Render
* Google Cloud Run
* AWS
* Azure

For Streamlit deployment, add the Gemini API key through the platform's secret/environment variable management instead of uploading `.env`.

---

## 🎯 Project Goals

The main goal of ParkWise AI is to explore how Agentic RAG can improve conventional Retrieval-Augmented Generation by allowing an AI system to make autonomous decisions during retrieval.

The project demonstrates concepts including:

* Large Language Models
* Retrieval-Augmented Generation
* Agentic AI
* Semantic Search
* Embeddings
* Query Rewriting
* Tool Selection
* Evidence Evaluation
* Multi-step Reasoning
* Context Grounding

---

## 🔮 Future Improvements

Future versions may include:

* 🌐 Real-time National Park Service API integration
* 🌤️ Live weather information
* 🗺️ Interactive maps
* 🏕️ Real-time campground availability
* 🎫 Permit information
* 🥾 Trail recommendations
* 📍 Location-aware recommendations
* 🔎 Web-search agent
* 🧠 Multiple specialized agents
* 💾 Conversation memory
* 📊 Retrieval confidence scoring
* 📚 Source citations
* 🗃️ Larger national park knowledge base
* 🔄 Hybrid vector + keyword search
* 🧪 Automated RAG evaluation

---

## 📈 Future Multi-Agent Architecture

ParkWise could eventually evolve into a multi-agent system:

```text
                    User
                      ↓
               Coordinator Agent
                      ↓
       ┌──────────────┼──────────────┐
       ↓              ↓              ↓
 Retrieval Agent  Weather Agent  Planning Agent
       ↓              ↓              ↓
 Documents          API           Reasoning
       └──────────────┼──────────────┘
                      ↓
                Verification Agent
                      ↓
                  Final Answer
```

---

## 🎓 Learning Outcomes

Through this project, I explored:

* Building an end-to-end RAG pipeline
* Creating vector embeddings
* Semantic document retrieval
* LLM integration
* Prompt engineering
* Agentic AI workflows
* Query planning
* Evidence validation
* Retry and fallback mechanisms
* Streamlit application development
* AI application deployment

---

## 👨‍💻 Author

**Shahrier Shanto**

Computer Science Student
San Francisco State University

Interested in:

* Software Engineering
* Artificial Intelligence
* Machine Learning
* Generative AI
* Agentic AI
* Full-Stack Development

---

## ⭐ Support

If you found this project interesting, consider giving the repository a ⭐.

---

## 📄 License

This project is intended for educational and research purposes.
