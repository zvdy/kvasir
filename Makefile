.PHONY: build build-gpu-tests fmt fmt-check lint test-cpu test-gpu kernels deny doc ci ci-local check-agent-pr doctor clean bench bench-kernels bench-forward bench-baseline ref

CARGO ?= cargo
REF_MODEL ?= models/Llama-3.2-1B
REF_OUT ?= ref/out
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu

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

build-gpu-tests:
	$(CARGO) test --workspace --release --features gpu-tests --no-run

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

ci: fmt-check lint test-cpu build-gpu-tests deny doc

ci-local: ci test-gpu

check-agent-pr:
	@scripts/check-agent-pr.sh

doctor:
	@scripts/doctor.sh

ref:
	@if [ ! -f $(REF_MODEL)/model.safetensors ]; then \
		echo "ref: weights absent ($(REF_MODEL)/model.safetensors); skipping (backlog ENV-weights)"; exit 0; fi; \
	set -e; \
	[ -x ref/.venv/bin/python ] || python3 -m venv ref/.venv; \
	ref/.venv/bin/pip install -q --extra-index-url $(TORCH_INDEX) -r ref/requirements.txt; \
	ref/.venv/bin/python ref/dump_logits.py --model $(REF_MODEL) --out $(REF_OUT)

clean:
	$(CARGO) clean
