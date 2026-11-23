# Installation Guide

## Prerequisites
- **Python 3.10+** (recommended via `conda` or `venv`).
- **Node 20+** and `npm` (or `pnpm`) for the frontend.
- **Git** for version control.
- **Docker** (optional, for containerised deployment).

## Backend setup
```bash
# Clone the repository
git clone https://github.com/your-org/DermaAI.git
cd DermaAI

# Create a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## Frontend setup
```bash
cd frontend
npm install   # or pnpm install if you prefer
npm run dev   # starts Vite dev server at http://localhost:5173
```

The frontend expects the backend at `http://localhost:8000/api`. Adjust the `API_BASE` constant in `src/components/UploadImage.tsx` if you change the address.

## Optional containerised run
```bash
docker compose up --build
```
This will start both backend and frontend containers and expose them on ports **8000** (API) and **5173** (UI).

---
*For a clean environment, delete the `venv` folder and reinstall from `requirements.txt`.*