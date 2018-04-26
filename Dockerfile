FROM sdavidhdez/jupyterlab:latest
LABEL maintainer="Simon D. Hernandez <simondhdez@totum.one>"

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gnupg \
    && curl -sL https://deb.nodesource.com/setup_8.x | bash - \
    && apt-get install -y --no-install-recommends \
                nodejs \
                build-essential \
    # && apt-get remove --purge curl gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN jupyter labextension install jupyterlab_vim \
    && chown someuser:someuser -R /home/someuser/*

RUN cd ~ && pip --no-cache-dir install nltk
RUN pip --no-cache-dir install whoosh
RUN pip --no-cache-dir install numpy
RUN pip --no-cache-dir install python-crfsuite

USER someuser
