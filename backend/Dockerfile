# backend/Dockerfile
FROM rust:latest

WORKDIR /app

# Install cargo-watch for hot-reloading
RUN cargo install cargo-watch

# Expose the Actix-Web port
EXPOSE 8080

# We don't set a CMD here because we will define it in docker-compose.yml
