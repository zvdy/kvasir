.PHONY: build test test-lib kernels lint fmt check clean

build:
	cargo build --release

test:
	cargo test --release

test-lib:
	cargo test --lib

kernels:
	cargo test --release --test kernels -- --nocapture --test-threads=1

lint:
	cargo clippy --all-targets -- -D warnings

fmt:
	cargo fmt --all

check: fmt lint test

clean:
	cargo clean
