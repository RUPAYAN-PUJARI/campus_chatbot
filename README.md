# BPPIMT Campus Chatbot

An AI-powered campus assistant for **BP Poddar Institute of Management & Technology**. Students and staff can ask natural language questions about faculty, placements, scholarships, and campus resources — with support for both text queries and face-photo search.

---

## Features

- **Faculty Search** — semantic search across 104 faculty profiles; filter by name, department, and designation
- **Face Recognition Search** — upload a photo to identify a faculty member via FaceNet embeddings
- **Placement Records** — query 900+ placement records by company, discipline, year, or student name; paginated results
- **Scholarship Discovery** — find matching schemes by eligibility, state, income, and target group
- **Campus Web Q&A** — answers sourced from scraped BPPIMT website content (admissions, timetable, facilities)
- **LLM-Powered Responses** — Groq GPT-OSS-20B generates natural language answers from retrieved context

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask 3.0.3 |
| Vector Search | FAISS (IndexFlatIP) |
| Text Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| Face Embeddings | FaceNet-PyTorch (MTCNN + InceptionResnetV1, 512-dim) |
| LLM | Groq API — GPT-OSS-20B (`openai/gpt-oss-20b`) |
| Web Scraping | BeautifulSoup4 |
| Frontend | React 18 (CDN, no build step), custom CSS |
| Image Processing | Pillow |
| Numerical | NumPy 1.26.4 |

---

## Project Structure

```
campus_chatbot/
├── main.py                           # CLI: build indexes, run CLI search
├── requirements.txt
├── .env                              # GROQ_API_KEY, GROQ_MODEL, GROQ_REASONING_EFFORT

├── backend/
│   ├── app.py                        # Flask server — routes, intent detection, orchestration
│   └── services/
│       └── llm_service.py            # Groq LLM calls (response generation + filter parsing)

├── frontend/
│   └── web/
│       ├── index.html                # App shell + all CSS
│       └── app.js                    # React UI — chat interface, result renderers

├── embeddings/                       # Embedding generation scripts
│   ├── generate_text_documents.py    # CSV/JSON → document objects
│   ├── generate_text_embeddings.py   # Documents → .npy embedding arrays
│   ├── generate_face_embeddings.py   # Faculty photos → face embeddings
│   ├── generate_placement_documents.py
│   └── generate_scholarship_documents.py

├── vector_db/                        # FAISS index builders
│   ├── build_faiss_index.py
│   ├── build_faiss_face_index.py
│   ├── faiss_text.index              # Faculty search index
│   ├── faiss_placement.index
│   ├── faiss_scholarship.index
│   ├── faiss_face.index
│   └── metadata.pkl

├── retrieval/                        # Domain-specific search logic
│   ├── text_search.py                # FAISS + keyword/attribute filtering
│   ├── face_search.py                # Face detection + FAISS similarity
│   └── web_search.py                 # Web scraping + FAISS

├── utils/
│   ├── csv_utils.py
│   ├── image_utils.py
│   ├── face_utils.py
│   ├── metadata_utils.py
│   └── logging_utils.py

├── data/
│   ├── faculty_list_cleaned.csv      # 104 faculty records
│   ├── placement_details.csv         # 900+ placement records
│   └── scholarship_details.json      # Scholarship schemes + eligibility

├── processed/                        # Auto-generated document JSON files
├── faculty_images/                   # Faculty photos (for face recognition)
└── scholarship_images/               # Scholarship provider logos
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
GROQ_REASONING_EFFORT=low
```

### 3. Build FAISS indexes (one-time)

```bash
# Build all text indexes (faculty, placement, scholarship, web)
python main.py build

# Build face recognition index separately (requires faculty_images/)
python main.py build-faces
```

### 4. Run the server

```bash
python backend/app.py
```

The app will be available at `http://localhost:5000`.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/assistant/text` | Main chat endpoint — routes to faculty/placement/scholarship/web |
| `POST` | `/api/assistant/image` | Face photo upload + conversational response |
| `POST` | `/api/search/text` | Direct text search with filters |
| `POST` | `/api/search/image` | Direct face recognition search |
| `GET` | `/api/filters` | Available departments and designations |
| `GET` | `/images/<filename>` | Serve faculty photos |
| `GET` | `/scholarship-images/<filename>` | Serve scholarship logos |

---

## How It Works

### Intent Detection

Every incoming query is classified before retrieval:

| Intent | Trigger Keywords |
|---|---|
| Placement | "placement", "placed", "company", "on campus", "off campus" |
| Scholarship | "scholarship", "scheme", "grant", "fee waiver" |
| Web | "admission", "course", "timetable", "facility", "hostel", "library" |
| Faculty (default) | name entities, designation/department terms |

### Query Pipeline

```
User Query (text or image)
        ↓
  Intent Detection
        ↓
  ┌─────┬──────────┬─────────────┬─────┐
  ↓     ↓          ↓             ↓     ↓
Faculty Placement Scholarship  Web  Face
  ↓     ↓          ↓             ↓     ↓
FAISS FAISS+CSV  FAISS+JSON   Scrape FAISS
Search Filter    Filter       +FAISS  Match
  └─────┴──────────┴─────────────┴─────┘
        ↓
  Groq LLM — generates natural language response from retrieved context
        ↓
  JSON response { answer, results, meta }
        ↓
  React UI renders chat bubble + structured result cards
```

### CLI Usage

```bash
# Run a search directly from terminal
python main.py search --query "machine learning faculty" --top-k 5
```

---

## Data Summary

| Dataset | Records | Key Fields |
|---|---|---|
| Faculty | 104 | Name, Department, Designation, Degree, University, Experience, Specialization |
| Placement | 900+ | Student, Enrollment No., Company, Discipline, Year, On/Off Campus, Academic Year |
| Scholarship | Variable | Title, Provider, State, Type, Eligibility (income/community), Benefits |
| Web Documents | Dynamic | Scraped pages from bppimt.ac.in |

---

## Frontend

The React frontend (`frontend/web/`) requires no build step — it loads React 18 from CDN.

**UI Layout:**
- **Left panel** — chat thread; user messages on the right, assistant on the left
- **Right sidebar** — sticky filters and usage tips
- **Bottom bar** — floating search bar with text input and image upload

**Result Rendering:**
- **Faculty** — photo card with name, designation, department, degree, experience
- **Placement** — scrollable table with student, employer, discipline, year, campus type
- **Scholarship** — expandable cards with eligibility criteria, benefits, and target groups
- **Web** — link cards with title, snippet, and PDF attachments
