# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test, and Development Commands

### Python (core package)

```bash
uv sync --group dev                                    # provision dev tooling (once per env)
uv run --no-sync pre-commit install                    # install git hooks (once per env)
uv run --no-sync pytest -v                             # full test suite
uv run --no-sync pytest tests/store/ -v                # run one test directory
uv run --no-sync pytest -k "test_enqueue" -v           # run tests matching a name pattern
uv run --no-sync pytest -m "not mongo and not gpu" -v  # skip optional markers
uv run --no-sync pyright                               # strict type checking (full, slow)
uv run --no-sync pyright -p pyrightconfig.fast.json     # fast type checking subset
uv run --no-sync pre-commit run --all-files --show-diff-on-failure  # formatting (black + isort)
uv run --no-sync mkdocs build --strict                 # validate documentation
```

When `uv run` errors with `Permission denied` under `~/.cache`, prefix with: `UV_CACHE="$(pwd)/.cache_uv" XDG_CACHE_HOME="$(pwd)/.cache_xdg"`.

Always commit the refreshed `uv.lock` when dependencies change. Mention optional dependency groups (verl, apo, gpu, agents, langchain, etc.) in PR notes.

### Dashboard (React/TypeScript UI)

```bash
cd dashboard && npm install                            # install dependencies
npm run dev                                            # local dev server (Vite)
npm run build                                          # typecheck + production build
npm run typecheck                                      # TypeScript only (tsc --noEmit)
npm run vitest                                         # unit tests (Vitest)
npm run vitest-storybook                               # Storybook interaction tests
npm run storybook                                      # Storybook dev server on :6006
npm run eslint                                         # lint
npm run stylelint                                      # CSS lint
npm run prettier                                       # format check
```

### Pyright fast vs full

`pyrightconfig.fast.json` excludes heavy optional modules: `verl/`, `instrumentation/`, `algorithm/apo/`, `algorithm/verl/`, `cli/vllm.py`, `store/mongo.py`, `store/collection/mongo.py`, `tracer/weave.py`. Use it for quick iteration; run the full config before pushing.

## Architecture Overview

Agent Lightning is an RL training framework for AI agents. The core loop:

1. **Algorithm** enqueues `Rollout` work items into the `LightningStore`
2. **Runner** dequeues rollouts, wraps execution in a **Tracer** context, and invokes the user's **LitAgent**
3. Agent executes (calling LLMs, tools, etc.) — spans are collected automatically via **instrumentation** or manually via **emitters**
4. Runner finalizes the `Attempt`, writing spans back to the store
5. **Adapter** transforms `List[Span]` → training data (e.g., `List[Triplet]` of prompt/response/reward)
6. Algorithm reads transformed traces and updates resources (prompt templates, model weights)
7. Loop repeats with updated resources

The **Trainer** wires everything together and delegates process orchestration to an **ExecutionStrategy**.

### Key Modules (`agentlightning/`)

| Module | Role |
|--------|------|
| `types/` | Shared Pydantic models: `Span`, `Rollout`, `Attempt`, `Triplet`, `Hook`, `Resource`, `LLM`, `ProxyLLM`, `PromptTemplate`, `NamedResources`, `Dataset` |
| `store/` | `LightningStore` ABC with `InMemoryLightningStore`, `MongoLightningStore`, and HTTP client/server (`LightningStoreClient`/`LightningStoreServer`) |
| `algorithm/` | `Algorithm` base class; implementations: `Baseline`, `ApoAlgorithm`, `VerlInterface`; `@algorithm` decorator |
| `tracer/` | `Tracer` ABC managing trace context and span collection; implementations: `AgentOpsTracer`, `OtelTracer`, `DummyTracer`, `WeaveTracer` |
| `emitter/` | First-party span creation: `emit_reward()`, `emit_message()`, `emit_object()`, `emit_exception()`, `operation()` context manager |
| `instrumentation/` | Auto-instruments third-party libraries (AgentOps, LiteLLM) to produce OTel spans without code changes |
| `runner/` | `Runner` ABC; `LitAgentRunner` polls store, manages tracer lifecycle, heartbeats, and hook invocation |
| `litagent/` | `LitAgent[T]` user-facing agent base class; `@litagent` decorator |
| `adapter/` | `TraceAdapter` converts `List[Span]` to training data; `TracerTraceToTriplet` is the default |
| `execution/` | `ExecutionStrategy` ABC; `ClientServerExecutionStrategy` ("cs", default, multi-process via HTTP) and `SharedMemoryExecutionStrategy` ("shm", single-process threads) |
| `trainer/` | `Trainer` orchestrator: resolves components via `ComponentSpec`, calls `fit()` or `dev()` |
| `cli/` | `agl` CLI entry point (`project.scripts.agl`): subcommands `vllm`, `store`, `prometheus`, `agentops` |
| `llm_proxy.py` | LiteLLM-based proxy for routing LLM calls with rollout-aware endpoints |
| `config.py` | `TrainerConfig` and component configuration |
| `env_var.py` | Environment variable constants |

### Three Span Layers

- **`instrumentation/`** — auto-patches external libraries (LiteLLM, AgentOps) so their calls emit OTel spans automatically
- **`tracer/`** — manages trace context scope and the span collection pipeline that writes to `LightningStore`
- **`emitter/`** — explicit helpers for user code to manually create annotation/reward/message spans

### Execution Strategies

- **`ClientServerExecutionStrategy`** ("cs", default) — algorithm runs in-process with an HTTP store server; runners spawn as separate OS processes connecting via `LightningStoreClient`. Supports distributed deployment via `AGL_CURRENT_ROLE` env var (`algorithm`, `runner`, `both`).
- **`SharedMemoryExecutionStrategy`** ("shm") — everything runs in one process with threads; store wrapped in `LightningStoreThreaded`. Useful for debugging.

### Store Collection Abstraction

`store/collection/` provides a pluggable data-structure layer underneath the store. Each store operation maps to a collection primitive:

- **`ListBasedCollection`** / **`MongoBasedCollection`** — ordered document storage with filtering
- **`DequeBasedQueue`** / **`MongoBasedQueue`** — FIFO work queues
- **`DictBasedKeyValue`** / **`MongoBasedKeyValue`** — key-value lookups

The in-memory variants use Python built-ins; Mongo variants use the same `AsyncMongoClient`. Both implement the same ABCs, so store tests are parametrized over both backends.

### Public API

The entire public API is re-exported from a flat `agentlightning` namespace via `__init__.py` wildcard imports. Each submodule's `__all__` controls what gets re-exported. Deprecated symbols still importable but marked: `AgentLightningClient`, `DevTaskLoader`, `AgentLightningServer`, `configure_logger` — avoid these in new code; use `Trainer` and `setup_logging` instead.

## Project Structure

- `agentlightning/` — core package (adapters, execution, training loop, tracer, reward, CLI)
- `tests/` — mirrors `agentlightning/` structure; store fixtures in `tests/store/conftest.py`
- `examples/` — self-contained runnable workflows (apo, azure, calc_x, chartqa, claude_code, minimal, rag, spider, tinker, unsloth)
- `contrib/` — community/third-party incubator: `contrib/recipes/<topic>/` for cookbooks, `contrib/agentlightning/contrib/<feature>/` for runtime extensions, `contrib/scripts/` for supporting automation
- `dashboard/` — React 19 / Redux Toolkit / Mantine 8 UI with 5 views: Rollouts, Traces, Workers, Resources, Settings. Tested with Vitest + Storybook + Playwright.
- `docs/` — mkdocs site (how-to guides, tutorials, deep-dives, algorithm-zoo); navigation in `mkdocs.yml`
- `scripts/` — release/CI/dataset automation

## Coding Conventions

- Python >= 3.10, 120-char lines, 4-space indent, Black + isort (`black` profile)
- `snake_case` for modules/functions/variables; `PascalCase` for classes and React components; lowercase-hyphenated for CLI flags, branch names, TypeScript filenames
- Exhaustive type hints enforced by pyright (strict mode). Prefer `agentlightning.types` dataclasses/Pydantic models.
- Google-style docstrings; mkdocs `[][]` cross-references; single backticks for inline code
- Logging via `logging.getLogger(__name__)` with appropriate DEBUG/INFO/WARNING/ERROR levels

## Testing

- Mirror runtime directories under `tests/` with matching filenames
- Pytest markers: `openai`, `gpu`, `agentops`, `weave`, `llmproxy`, `mongo`, `store`, `prometheus`, `utils`, `langchain`
- Key shared fixtures live in `tests/store/conftest.py` (there is no top-level `tests/conftest.py`): `inmemory_store`, `mongo_store`, `store_fixture` (parametrized over both), `mock_readable_span`, `fake_time`
- MongoDB tests require `AGL_TEST_MONGO_URI` (default: `mongodb://localhost:27017/?replicaSet=rs0`)
- Async store tests use `pytest-asyncio`; sync fixtures are the norm elsewhere
- Favor real stores/spans/agents over mocks; parametrize test cases

## Dependency Groups

The project uses `uv` with many optional dependency groups in `pyproject.toml`:

| Group | When to use |
|-------|-------------|
| `dev` | Always, for development tooling |
| `torch-stable` / `torch-legacy` | GPU training (mutually exclusive) |
| `torch-gpu-stable` / `torch-gpu-legacy` | GPU with flash-attn + verl |
| `agents` | AutoGen, OpenAI Agents, Anthropic, CrewAI, SWE-bench |
| `langchain` | LangChain/LangGraph (conflicts with torch-legacy) |
| `mongo` | MongoDB store backend |
| `apo` | Automatic Prompt Optimization |
| `trl` | TRL/Unsloth fine-tuning |
| `tinker` | Tinker RL backend (requires Python >= 3.11) |
| `image` | Multi-modal (Pillow, datasets, qwen-vl-utils) |

Groups have declared conflicts (e.g., `core-legacy` vs `core-stable`, `torch-cpu` vs `torch-cu128`, `langchain` vs `torch-legacy`). Check `[tool.uv]` in `pyproject.toml`.

## Example Contributions

- Each example needs a README with smoke-test instructions and an "Included Files" section
- Self-contained modules with a module-level docstring describing CLI usage
- CI workflow per example: `examples-<name>.yml` in `.github/workflows/`, registered in `badge-<name>.yml`, `badge-examples.yml`, and `badge-latest.yml`

## Commits and Pull Requests

- Branch from `main`: `feature/<slug>`, `fix/<slug>`, `docs/<slug>`, `chore/<slug>`
- Imperative commit messages; reference issues with `Fixes #123`
- Run pre-commit + relevant pytest/pyright/mkdocs before pushing
- PR descriptions: summarize intent, list verification commands, note dependency or docs-navigation changes
