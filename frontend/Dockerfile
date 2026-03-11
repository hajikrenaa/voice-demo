FROM nginx:alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Add custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy static frontend files
COPY . /usr/share/nginx/html

# Remove Dockerfile and nginx.conf from served files
RUN rm -f /usr/share/nginx/html/Dockerfile /usr/share/nginx/html/nginx.conf

EXPOSE 80
