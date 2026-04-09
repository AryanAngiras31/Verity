# frontend/Dockerfile
FROM node:20-slim

WORKDIR /app

# We don't copy package.json right away because we will mount the entire
# frontend directory as a volume in docker-compose.
# This ensures you can edit files on your host and see them hot-reload.

EXPOSE 5173

# When the container starts, ensure packages are installed and start the dev server
CMD ["sh", "-c", "npm install && npm run dev -- --host 0.0.0.0"]
