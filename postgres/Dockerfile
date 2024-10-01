# Use the base image from ParadeDB which includes pg_search
FROM paradedb/paradedb:latest

# Set Postgres version (assuming PG_MAJOR is 16 as in your pgvector Dockerfile)
ARG PG_MAJOR=16

# Fix permissions for /var/lib/apt/lists
USER root
RUN mkdir -p /var/lib/apt/lists/partial && chmod -R 777 /var/lib/apt/lists/partial

# Install git to clone pgvector
RUN apt-get update && apt-get install -y git

# Clone pgvector source code
RUN git clone https://github.com/pgvector/pgvector.git /tmp/pgvector

# Install dependencies, compile and install pgvector, clean up
RUN apt-get update && \
    apt-mark hold locales && \
    apt-get install -y --no-install-recommends build-essential postgresql-server-dev-$PG_MAJOR && \
    cd /tmp/pgvector && \
    make clean && \
    make OPTFLAGS="" && \
    make install && \
    mkdir /usr/share/doc/pgvector && \
    cp LICENSE README.md /usr/share/doc/pgvector && \
    rm -r /tmp/pgvector && \
    apt-get remove -y build-essential postgresql-server-dev-$PG_MAJOR && \
    apt-get autoremove -y && \
    apt-mark unhold locales && \
    rm -rf /var/lib/apt/lists/*

# Clean up
RUN apt-get remove -y git && apt-get autoremove -y

# Expose Postgres default port
EXPOSE 5432
