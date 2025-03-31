# Backend Setup (Python)

## 1. Create and activate the environment

```bash
conda env create -n chat-bot -f environment.yml
conda activate chat-bot
```

## 2. Sync dependencies
```bash
uv pip sync requirements.txt
```

If you changed something in the setug.cfg, run:

```bash
uv pip compile setup.cfg -o requirements.txt
uv pip sync requirements.txt
```

## 3. Start the backend
```bash
uvicorn backend.app:app --reload
```

# Frontend Setup (React)

## 1. Navigate to the frontend folder and install dependencies

```bash
cd frontend
npm install
```

## 2. Run the frontend development server

```bash
npm run dev
```

This starts the React interface at http://localhost:5173. Make sure the backend is running at http://localhost:8000.


# API Endpoints

- POST /chat — receives { "text": "..." }, returns { "response": "..." }

- POST /debug_prompt — returns the full prompt (context + memory + question)

- GET /memory — returns full conversation history