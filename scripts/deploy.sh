#!/bin/bash
# Deploy latest changes and restart containers
# Usage: ./deploy.sh

echo "Pulling latest changes..."
git pull

echo "Building and restarting containers..."

# Check if production override exists
if [ -f "docker-compose.prod.yml" ]; then
    echo "Production config found. Using docker-compose.prod.yml for SSL..."
    sudo docker compose down
    sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml build
    sudo docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
else
    echo "No production config found. Using default docker-compose.yml..."
    sudo docker compose down
    sudo docker compose build
    sudo docker compose up -d
fi

echo "Deployment complete! Checking status:"
sudo docker ps
