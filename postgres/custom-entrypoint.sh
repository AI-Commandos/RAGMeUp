#!/bin/bash
set -e

# Run the original entrypoint script
source /usr/local/bin/docker-entrypoint.sh

# Create the pgvector extension in the 'langchain' database
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "langchain" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS vector;
EOSQL

# Run the main command (usually postgres)
exec "$@"
