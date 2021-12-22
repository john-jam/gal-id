#!/bin/bash

INIT_FLAG_FILE=/root/.init-gal-id
GAL_ID_PROJECT_PATH=/deploy/gal-id
GAL_ID_DOCKER_COMPOSE_URL=https://raw.githubusercontent.com/john-jam/gal-id/main/docker-compose.yml
GAL_ID_RUN_SCRIPT_URL=https://raw.githubusercontent.com/john-jam/gal-id/main/gcp/run.sh
VM_USER=john

if [[ ! -f ${INIT_FLAG_FILE} ]]; then
    # Init the machine for the first time

    apt update -y
    apt upgrade -y
    apt install -y \
        ca-certificates \
        curl \
        gnupg \
        haveged \
        lsb-release

    # Install docker
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt update -y
    apt install -y \
        docker-ce \
        docker-ce-cli \
        containerd.io

    # Post-install steps for docker (not recommended for production)
    groupadd docker || true
    usermod -aG docker ${VM_USER}

    # Install docker-compose
    curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose

    # Clone the docker-compose file
    mkdir -p ${GAL_ID_PROJECT_PATH}
    wget ${GAL_ID_DOCKER_COMPOSE_URL} -P ${GAL_ID_PROJECT_PATH}

    # Clone the run.sh script
    wget ${GAL_ID_RUN_SCRIPT_URL} -P ${GAL_ID_PROJECT_PATH}
    chmod +x ${GAL_ID_PROJECT_PATH}/run.sh

    # Give the vm user permissions to the project folder
    chown -R ${VM_USER}:${VM_USER} ${GAL_ID_PROJECT_PATH}

    # Create a flag file to avoid running this initialization again
    touch ${INIT_FLAG_FILE}

    echo 'Instance initialized'
else
    echo 'Instance already initialized'
fi
