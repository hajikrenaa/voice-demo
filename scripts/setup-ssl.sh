#!/bin/bash
# Install SSL Certificate for Domain

if [ -z "$1" ]; then
    echo "Usage: ./setup-ssl.sh [YOUR_DOMAIN]"
    exit 1
fi

DOMAIN=$1
EMAIL="akashimman@gmail.com"

echo "Setting up SSL for domain: $DOMAIN"

# 1. Install Certbot
echo "Installing Certbot..."
sudo snap install --classic certbot
sudo ln -sf /snap/bin/certbot /usr/bin/certbot

# 2. Stop Docker to free port 80
echo "Stopping Docker containers to free port 80..."
sudo docker compose down

# 3. Request SSL Certificate
echo "Requesting SSL Certificate from Let's Encrypt..."
sudo certbot certonly --standalone -d $DOMAIN -m $EMAIL --agree-tos -n

if [ ! -d "/etc/letsencrypt/live/$DOMAIN" ]; then
    echo "Certbot failed to obtain SSL certificate. Double check your DNS records."
    sudo docker compose up -d
    exit 1
fi

echo "SSL Certificate procured successfully!"

# 4. Generate Production Docker Compose Override
echo "Generating docker-compose.prod.yml..."
cat <<EOF > docker-compose.prod.yml
services:
  frontend:
    volumes:
      - /etc/letsencrypt/live/$DOMAIN/fullchain.pem:/etc/nginx/ssl/nginx-selfsigned.crt:ro
      - /etc/letsencrypt/live/$DOMAIN/privkey.pem:/etc/nginx/ssl/nginx-selfsigned.key:ro
EOF

echo "Done generating production override."

# 5. Restart with new config
echo "Restarting application with secure configuration..."
sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

echo ""
echo "Installation complete!"
echo "Your app is now securely served at https://$DOMAIN"
