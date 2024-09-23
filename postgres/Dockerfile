FROM pgvector/pgvector:pg16

USER root
RUN apt-get update && apt-get install -y libicu74
COPY custom_postgresql.conf /etc/postgresql/postgresql.conf
USER postgres
CMD ["postgres", "-c", "config_file=/etc/postgresql/postgresql.conf"]
