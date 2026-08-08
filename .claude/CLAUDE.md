# Coding Workflow

A working agreement for building projects with an AI coding assistant. Read this first; follow it unless I explicitly override a step.

---

## Hard Rules (never violate)

- **NEVER write secrets into any file that could be committed** — no API keys, tokens, passwords, connection strings. Use environment variables or a secrets manager, and reference them by name only.
- **Never start implementing before the plan is approved.** Stop at each approval gate and wait.
- **Right-size the process.** This is a full pipeline, not a mandatory one. A throwaway script doesn't need Kube manifests; a one-file batch job doesn't need a service layer. When a phase doesn't apply, say so explicitly and skip it — don't silently drop it, and don't perform it as ceremony.
- **Don't gold-plate.** Build what the spec asks for. Flag tempting additions as "out of scope unless you want it" rather than implementing them unasked.
- **Surface decisions, don't bury them.** When you hit a fork (library choice, schema shape, tradeoff), name the options and your recommendation before committing to one.
- write code using the best state of the art practices, keep in mind maintainability, code deduplication, abstractions etc... 
- keep code concise, as little as possible, if you notice delete code which is not used any more or is redundant, keep the repo clean and tidy, all the files and configs must be used
- do not do excessive comments, just the essence, cover intention why and not so much what in the comments (the code must be written so that WHAT it is doing is clear from the code)

---

## The Flow

```
Spec → Architecture → Plan → [ Implement → Error-check → Test ]×components → Refactor + Tooling → Build → Deploy → Security Review
```

The bracketed loop repeats per component. Everything before it is design; everything after is hardening.

---

## 1. Spec — `docs/tmp/spec.md`

Write before any design. Keep it to one page.

- **Goal** — one sentence: the problem solved.
- **Users** — who runs this and in what context.
- **Core features** — numbered must-haves.
- **Non-goals** — explicitly out of scope (this section prevents scope creep later).
- **Success criteria** — measurable. How do we *know* it works? What's the acceptance test?
- **Constraints** — language, runtime, budget (incl. paid-API spend), data sensitivity, deadlines.

> **Gate:** I approve the spec before architecture begins.

---

## 2. Architecture — `docs/tmp/architecture.md`

Design only. No code.

- Component breakdown + data flow (a diagram or a clear list).
- Stack choices, each with a one-line *why* and the main alternative rejected.
- Data models / schemas.
- Interfaces and contracts between components (function signatures, API shapes, file formats).
- External dependencies and the reason each one earns its place.
- The two or three decisions most likely to be wrong, called out for scrutiny.

> **Gate:** I approve the architecture before planning.

---

## 3. Plan — per component

Break the architecture into components in dependency order. For each:

- Ordered implementation steps.
- Inputs, outputs, side effects.
- The specific error cases this component must handle.
- Its definition of done.

> **Gate:** I approve the plan before any code is written.

---

## 4–6. The Component Loop

For **each** component, in order, complete all three before touching the next one.

### 4. Implement
- Small, single-purpose functions.
- No magic values — named constants or config.
- Secrets only via environment variables.
- Commit in small, coherent chunks with clear messages.

### 5. Error handling — checklist
- [ ] Every failure path handled explicitly; no silent `except: pass`.
- [ ] External calls have timeouts; retries where a transient failure is plausible.
- [ ] Errors carry enough context to debug (what failed, on what input).
- [ ] Edge cases covered: empty, null, malformed, oversized, concurrent.
- [ ] Failures degrade gracefully where the design allows.

### 6. Tests
- [ ] Unit tests for the core logic.
- [ ] Integration tests at every external boundary (DB, API, filesystem, queue).
- [ ] The edge cases from the error checklist are actually tested, not just handled.
- [ ] Tests run in CI with no manual setup.
- [ ] All tests green before moving on.

> **Gate:** A component isn't "done" until its tests pass. Don't start the next one until it is.

---

## 7. Refactor + Tooling

Once all components are built and green:

- Add a linter and formatter (`ruff`/`black`, `eslint`/`prettier`, etc.) and wire them into a pre-commit hook.
- Remove dead code, stray TODOs, debug prints.
- Unify naming and module structure across the codebase.
- Re-run the full test suite after refactoring.

---

## 8. Production Build

- Reproducible dependency pinning (lockfile committed).
- Build/run scripts.
- Config separated by environment (dev / staging / prod), no hardcoded values.
- Versioned, deterministic build artifact.

*(Skip or trim if the deliverable is a script or notebook, not a deployed service.)*

---

## 9. Deployment

*(Only if it actually ships as a running service.)*

- `Dockerfile`: minimal base, non-root user, no dev deps in the final image, no secrets in layers.
- Local dev compose file if useful.
- K8s manifests: Deployment, Service, ConfigMap, Secret (names only — no values), resource requests/limits, HPA if needed.
- Health endpoints (`/healthz`, `/readyz`) wired to liveness/readiness probes.
- Document every runtime environment variable.

---

## 10. Security Review — checklist

- [ ] No secrets in code, config, logs, git history, or Docker layers.
- [ ] All external input validated and sanitized.
- [ ] Auth + authorization enforced on every protected path.
- [ ] Dependencies scanned for known CVEs (`pip-audit`, `npm audit`, Dependabot).
- [ ] Least privilege everywhere: containers, IAM, DB users.
- [ ] Sensitive data encrypted at rest and in transit.
- [ ] Rate limiting on public endpoints.
- [ ] Security headers / CORS / CSP set where applicable.
- [ ] Client-facing errors leak no internals.

---

## Approval Gates (summary)

| Gate | Must clear before |
|---|---|
| Spec approved | Architecture |
| Architecture approved | Planning |
| Plan approved | Writing code |
| Component tests green | Next component |
| Full suite green | Refactor / build |
| Build verified | Deployment configs |
| Security review passed | Ship |

When in doubt, stop and ask rather than assume.


## Progress Log — `docs/tmp/progress.md`

Single source of truth for project state across model handoffs. Update it at every gate and after every component. Paste it as context when starting a new session or switching models.

Keep it terse — state and decisions, not a diary.

### Format

**Phase status**
- [x] Spec — approved
- [x] Architecture — approved
- [ ] Component: CSV loader — implemented, tests pending
- [ ] Component: VLM verdict — not started

**Decision log** (append-only; one line each)
- Chose CLIP over 7B VLM — fits free Colab tier, accepts weaker reasoning.
- History overrides image only when image verdict is `not_enough_information` — per spec precedence rule.

**Open questions / blockers**
- Confirm exact `severity` vocabulary from problem_statement.md before finalizing output schema.


## Model Tier by Phase

**Guiding principle:** spend tokens in inverse proportion to volume and in direct proportion to blast radius. Low-volume, high-consequence phases (spec, architecture, security, test design) get the best model. The high-volume, low-consequence-per-unit phase (routine coding) is where a cheaper model pays off — and even there, escalate the few high-blast-radius pieces.

| Phase | Model | Why |
|---|---|---|
| Spec | **Top** | High leverage — errors propagate into everything downstream. |
| Architecture | **Top** | Same; load-bearing decisions are cheap to make, expensive to fix. |
| Planning | **Top** | The plan is what makes cheaper coding safe; invest here. |
| Implementation | **Mid** | Mechanical work against an approved plan. *Escalate to top for the hard parts: concurrency, tricky algorithms, anything the plan flagged as risky.* |
| Test design | **Top** | Deciding *what* could break is adversarial reasoning. |
| Test writing | **Mid** | Boilerplate once the cases are identified. |
| Refactor + tooling | **Mid** | Pattern work, low blast radius. |
| Build / deploy config | **Mid** | Mostly templated; escalate if the infra is novel. |
| Security review | **Top** | Adversarial, low-volume, catastrophic if missed. Never economize. |



