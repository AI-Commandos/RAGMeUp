FROM python:3.11.5

RUN apt-get -y update && apt-get -y install build-essential \
    libpq-dev libssl-dev

RUN mkdir /ragmeup
RUN mkdir /data
WORKDIR /ragmeup

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install gunicorn

EXPOSE 5000

COPY . .

ARG UNAME=interfaceuser
ARG UID=101
ARG GID=101

RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME

RUN chown -R $UID:$GID /ragmeup
RUN chown -R $UID:$GID /data

USER $UNAME
