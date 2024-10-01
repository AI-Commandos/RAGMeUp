# Start from the ParadeDB image
FROM paradedb/paradedb:latest

# Install build dependencies
USER root
RUN apt-get update && \
    apt-mark hold locales && \
    apt-get install -y --no-install-recommends \
        build-essential \
        postgresql-server-dev-16 \
        git

# Clone pgvector
RUN git clone https://github.com/pgvector/pgvector.git /tmp/pgvector

# Build and install pgvector
RUN cd /tmp/pgvector && \
    make clean && \
    make OPTFLAGS="" && \
    make install && \
    mkdir /usr/share/doc/pgvector && \
    cp LICENSE README.md /usr/share/doc/pgvector

# Clean up
RUN rm -r /tmp/pgvector && \
    apt-get remove -y build-essential postgresql-server-dev-16 git && \
    apt-get autoremove -y && \
    apt-mark unhold locales && \
    rm -rf /var/lib/apt/lists/*

# Switch back to postgres user
USER postgres

# Add pgvector to shared_preload_libraries
RUN echo "shared_preload_libraries = 'pg_search,vector'" >> /usr/share/postgresql/postgresql.conf.sample

# Use the default CMD from ParadeDB
CMD ["postgres"]
