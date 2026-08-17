.PHONY: help install backend frontend test clean docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make install    - Install backend and frontend dependencies"
	@echo "  make backend    - Run FastAPI backend locally with reload"
	@echo "  make frontend   - Run Vite React frontend locally"
	@echo "  make test       - Run backend test suite with pytest"
	@echo "  make docker-up  - Start containers using docker-compose"
	@echo "  make docker-down- Stop docker containers"
	@echo "  make clean      - Clean cache and temporary files"

install:
	pip install -r backend/requirements.txt
	cd frontend && npm install

backend:
	cd backend && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

test:
	cd backend && python -m pytest -v

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
