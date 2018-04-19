FROM sdavidhdez/jupyterlab:latest
LABEL maintainer="Simon D. Hernandez <simondhdez@totum.one>"

USER root

RUN cd ~ && pip --no-cache-dir install nltk
RUN pip --no-cache-dir install whoosh

USER someuser
