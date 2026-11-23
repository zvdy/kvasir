.PHONY: build fmt fmt-check lint test-cpu test-gpu kernels bench bench-kernels bench-forward bench-baseline deny doc ci ci-local doctor clean

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

BENCH = $(CARGO) bench -p kvasir --features gpu-tests --bench

bench: bench-kernels bench-forward

bench-kernels:
	$(BENCH) kernels -- --out $(CURDIR)/target/bench/kernels.json --compare $(CURDIR)/bench/kernels-baseline.json

bench-forward:
	$(BENCH) forward -- --out $(CURDIR)/target/bench/forward.json --compare $(CURDIR)/bench/forward-baseline.json

bench-baseline:
	$(BENCH) kernels -- --out $(CURDIR)/bench/kernels-baseline.json
	$(BENCH) forward -- --out $(CURDIR)/bench/forward-baseline.json

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
