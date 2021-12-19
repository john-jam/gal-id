#!/bin/bash

GAL_ID_PROJECT_PATH=/deploy/gal-id

# Run the project inside docker containers
cd ${GAL_ID_PROJECT_PATH} || exit 1
export GI_DASHBOARD_URL="http://gal-id.jonathanlanglois.fr:6006" && docker-compose up -d
echo 'Applications started'