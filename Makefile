.PHONY: up down build logs test \
        db-upgrade db-downgrade db-stamp db-current db-history db-revision

# ── Docker Compose ────────────────────────────────────────────────────────────

up:
	docker compose up --build -d

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	python3 -m pytest tests/ -v --timeout=120 -p no:warnings

# ── Database migrations (Alembic) ─────────────────────────────────────────────
# Targets run against the mapped host port (5433) so they work from your laptop.
# To run inside Docker instead: docker compose exec prompt-registry alembic <cmd>

DB_URL ?= postgresql://postgres:postgres@localhost:5433/aibackend

db-upgrade:
	DATABASE_URL=$(DB_URL) alembic upgrade head

db-downgrade:
	DATABASE_URL=$(DB_URL) alembic downgrade -1

# Mark an existing database as already at the latest revision without running DDL.
# Use this on databases initialised by init.sql before Alembic was introduced.
db-stamp:
	DATABASE_URL=$(DB_URL) alembic stamp head

db-current:
	DATABASE_URL=$(DB_URL) alembic current

db-history:
	DATABASE_URL=$(DB_URL) alembic history --verbose

# Usage: make db-revision message="add prompt tags index"
db-revision:
	DATABASE_URL=$(DB_URL) alembic revision -m "$(message)"
