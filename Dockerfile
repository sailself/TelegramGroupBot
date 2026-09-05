FROM rust:1.92-slim AS builder

WORKDIR /app

COPY Cargo.toml ./
COPY Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/telegram_group_helper_bot /app/telegram_group_helper_bot

VOLUME /app/data

ENV DATABASE_URL=sqlite:///app/data/bot.db?mode=rwc

CMD ["/app/telegram_group_helper_bot"]
