FROM rust:latest

WORKDIR /app

# 1. Install the hot-reloading tool
RUN cargo install cargo-watch

# 2. Copy ONLY the dependency manifests first
COPY Cargo.toml ./
# If you have a Cargo.lock, uncomment the line below:
# COPY Cargo.lock ./

# 3. Create a dummy src/main.rs to trick Cargo into downloading and compiling dependencies
RUN mkdir src && \
    echo "fn main() { println!(\"Dummy build\"); }" > src/main.rs && \
    cargo build && \
    rm -rf src

# 4. NOW copy your actual source code
COPY src ./src

# 5. Expose the Actix port
EXPOSE 8080

# (Optional but good practice) Set a default command if run without Compose
CMD ["cargo", "run", "--release"]
