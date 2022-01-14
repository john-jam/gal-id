# Create instance on GCP
gcloud compute instances create gal-id \
  --project=gal-id \
  --zone=europe-west1-c \
  --machine-type=n2-standard-8 \
  --network-interface=network-tier=PREMIUM,subnet=default \
  --metadata=startup-script=\#\!/bin/bash$'\n'$'\n'INIT_FLAG_FILE=/root/.init-gal-id$'\n'GAL_ID_PROJECT_PATH=/deploy/gal-id$'\n'GAL_ID_DOCKER_COMPOSE_URL=https://raw.githubusercontent.com/john-jam/gal-id/main/docker-compose.yml$'\n'GAL_ID_RUN_SCRIPT_URL=https://raw.githubusercontent.com/john-jam/gal-id/main/gcp/run.sh$'\n'VM_USER=john$'\n'$'\n'if\ \[\[\ \!\ -f\ \$\{INIT_FLAG_FILE\}\ \]\]\;\ then$'\n'\ \ \ \ \#\ Init\ the\ machine\ for\ the\ first\ time$'\n'$'\n'\ \ \ \ apt\ update\ -y$'\n'\ \ \ \ apt\ upgrade\ -y$'\n'\ \ \ \ apt\ install\ -y\ \\$'\n'\ \ \ \ \ \ \ \ ca-certificates\ \\$'\n'\ \ \ \ \ \ \ \ curl\ \\$'\n'\ \ \ \ \ \ \ \ gnupg\ \\$'\n'\ \ \ \ \ \ \ \ haveged\ \\$'\n'\ \ \ \ \ \ \ \ lsb-release$'\n'$'\n'\ \ \ \ \#\ Install\ docker$'\n'\ \ \ \ curl\ -fsSL\ https://download.docker.com/linux/ubuntu/gpg\ \|\ sudo\ gpg\ --dearmor\ -o\ /usr/share/keyrings/docker-archive-keyring.gpg$'\n'\ \ \ \ echo\ \\$'\n'\ \ \ \ \ \ \"deb\ \[arch=\$\(dpkg\ --print-architecture\)\ signed-by=/usr/share/keyrings/docker-archive-keyring.gpg\]\ https://download.docker.com/linux/ubuntu\ \\$'\n'\ \ \ \ \ \ \$\(lsb_release\ -cs\)\ stable\"\ \|\ sudo\ tee\ /etc/apt/sources.list.d/docker.list\ \>\ /dev/null$'\n'\ \ \ \ apt\ update\ -y$'\n'\ \ \ \ apt\ install\ -y\ \\$'\n'\ \ \ \ \ \ \ \ docker-ce\ \\$'\n'\ \ \ \ \ \ \ \ docker-ce-cli\ \\$'\n'\ \ \ \ \ \ \ \ containerd.io$'\n'$'\n'\ \ \ \ \#\ Post-install\ steps\ for\ docker\ \(not\ recommended\ for\ production\)$'\n'\ \ \ \ groupadd\ docker\ \|\|\ true$'\n'\ \ \ \ usermod\ -aG\ docker\ \$\{VM_USER\}$'\n'$'\n'\ \ \ \ \#\ Install\ docker-compose$'\n'\ \ \ \ curl\ -L\ \"https://github.com/docker/compose/releases/download/1.29.2/docker-compose-\$\(uname\ -s\)-\$\(uname\ -m\)\"\ -o\ /usr/local/bin/docker-compose$'\n'\ \ \ \ chmod\ \+x\ /usr/local/bin/docker-compose$'\n'$'\n'\ \ \ \ \#\ Clone\ the\ docker-compose\ file$'\n'\ \ \ \ mkdir\ -p\ \$\{GAL_ID_PROJECT_PATH\}$'\n'\ \ \ \ wget\ \$\{GAL_ID_DOCKER_COMPOSE_URL\}\ -P\ \$\{GAL_ID_PROJECT_PATH\}$'\n'$'\n'\ \ \ \ \#\ Clone\ the\ run.sh\ script$'\n'\ \ \ \ wget\ \$\{GAL_ID_RUN_SCRIPT_URL\}\ -P\ \$\{GAL_ID_PROJECT_PATH\}$'\n'\ \ \ \ chmod\ \+x\ \$\{GAL_ID_PROJECT_PATH\}/run.sh$'\n'$'\n'\ \ \ \ \#\ Give\ the\ vm\ user\ permissions\ to\ the\ project\ folder$'\n'\ \ \ \ chown\ -R\ \$\{VM_USER\}:\$\{VM_USER\}\ \$\{GAL_ID_PROJECT_PATH\}$'\n'$'\n'\ \ \ \ \#\ Create\ a\ flag\ file\ to\ avoid\ running\ this\ initialization\ again$'\n'\ \ \ \ touch\ \$\{INIT_FLAG_FILE\}$'\n'$'\n'\ \ \ \ echo\ \'Instance\ initialized\'$'\n'else$'\n'\ \ \ \ echo\ \'Instance\ already\ initialized\'$'\n'fi$'\n' \
  --maintenance-policy=MIGRATE \
  --service-account=281168149935-compute@developer.gserviceaccount.com \
  --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
  --tags=gal-id,http-server \
  --create-disk=auto-delete=yes,boot=yes,device-name=gal-id,image=projects/ubuntu-os-cloud/global/images/ubuntu-2004-focal-v20211212,mode=rw,size=50,type=projects/gal-id/zones/europe-west1-b/diskTypes/pd-ssd \
  --no-shielded-secure-boot \
  --shielded-vtpm \
  --shielded-integrity-monitoring \
  --reservation-affinity=any

# Redirect ports on GCP
gcloud compute \
  --project=gal-id \
  firewall-rules create gal-id \
  --direction=INGRESS \
  --priority=1000 \
  --network=default \
  --action=ALLOW \
  --rules=tcp:8000,tcp:8501,tcp:6006 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=gal-id