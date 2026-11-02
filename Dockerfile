# Drop Chess: one container that serves the web client and runs the game server.
# Build: docker build -t drop-chess .   Run: docker run -p 8080:8080 drop-chess

# 1. Web client
FROM node:22-slim AS web
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# 2. Rust server (bookworm on both sides so the binary's glibc matches the runtime)
FROM rust:1-slim-bookworm AS server
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY engine ./engine
COPY server ./server
# pyengine is a workspace member (training bindings); only its manifest is needed here.
COPY pyengine ./pyengine
RUN cargo build --release --locked -p dropchess-server

# 3. Small runtime image
FROM debian:bookworm-slim
WORKDIR /app
COPY --from=server /app/target/release/server /app/server
COPY --from=web /app/web/dist /app/web/dist
ENV PORT=8080 DROPCHESS_WEB=/app/web/dist
EXPOSE 8080
CMD ["/app/server"]
