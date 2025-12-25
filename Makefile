.PHONY: build run stop restart logs clean help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build the Docker image
	docker-compose build

run: ## Start the application
	docker-compose up -d
	@echo "Application started at http://localhost:8501"

# run-with-db: ## Start the application with sample database
# 	docker-compose --profile with-sample-db up -d
# 	@echo "Application started at http://localhost:8501"
# 	@echo "Sample PostgreSQL available at localhost:5432"
# 	@echo "Connection: postgresql://demo:demo123@postgres-sample:5432/sampledb"

stop: ## Stop the application
	docker-compose down

restart: ## Restart the application
	docker-compose restart

logs: ## View application logs
	docker-compose logs -f natural-db-interface

logs-all: ## View all service logs
	docker-compose logs -f

shell: ## Open shell in the container
	docker-compose exec natural-db-interface /bin/bash

clean: ## Remove containers, volumes, and images
	docker-compose down -v
	docker rmi natural-database-interface-natural-db-interface 2>/dev/null || true

rebuild: clean build run ## Clean, rebuild, and run

ps: ## Show running containers
	docker-compose ps

dev: ## Run in development mode (with volume mount)
	docker-compose up