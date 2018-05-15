FROM sdavidhdez/ml-py3:latest
LABEL maintainer="Simon D. Hernandez <sdhdez@totum.one>"

ARG user=jupyterlab

USER root
COPY run_cmd.sh /
USER $user

CMD ["/run_cmd.sh"]
