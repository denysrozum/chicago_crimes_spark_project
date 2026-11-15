FROM apache/airflow:slim-2.10.5-python3.11

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.3.1.tgz \
      | tar -xz --strip-components=1 -C /usr/local/bin docker/docker && \
    chmod +x /usr/local/bin/docker && \
    rm -rf /var/lib/apt/lists/*
USER airflow

RUN pip install --no-cache-dir \
      --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.11.txt \
      psycopg2-binary==2.9.10
