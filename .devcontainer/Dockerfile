FROM nvcr.io/nvidia/tensorflow:24.06-tf2-py3

ARG UID
ARG GID
ARG USERNAME
ARG GROUPNAME

RUN apt update && apt install -y sudo

RUN groupadd -g ${GID} ${GROUPNAME}} -f && \
  useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME}

RUN echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers

USER ${USERNAME}

RUN pip install --upgrade pip
RUN pip install notebook
RUN pip install transformers torch bitsandbytes accelerate
