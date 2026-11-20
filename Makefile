.PHONY: build fmt fmt-check lint test-cpu test-gpu kernels deny doc ci ci-local doctor clean

CARGO ?= cargo

build:
	$(CARGO) build --release

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

lint:
	$(CARGO) clippy --workspace --all-targets -- -D warnings
	$(CARGO) clippy --workspace --all-targets --features gpu-tests -- -D warnings

test-cpu:
	$(CARGO) test --workspace

test-gpu:
	$(CARGO) test --workspace --release --features gpu-tests -- --test-threads=1

kernels:
	$(CARGO) test --release --features gpu-tests --test kernels -- --nocapture --test-threads=1

deny:
	$(CARGO) deny --all-features check

doc:
	RUSTDOCFLAGS=-Dwarnings $(CARGO) doc --workspace --no-deps

ci: fmt-check lint test-cpu deny doc

ci-local: ci test-gpu

doctor:
	@scripts/doctor.sh

clean:
	$(CARGO) clean
