# GitHub Actions CI/CD — Learning Guide

This guide explains every workflow in this repo, how Actions works conceptually,
and what to do to get everything running on GitHub.

---

## 1. What is CI/CD?

**CI — Continuous Integration**: automatically run checks (lint, tests) every
time someone pushes code. Catches bugs before they reach main.

**CD — Continuous Deployment**: automatically deploy working code to production
(e.g., HuggingFace Spaces) after it passes CI.

```
You push code
     │
     ▼
GitHub receives the push
     │
     ▼
GitHub Actions runner spins up (a fresh Ubuntu VM in the cloud)
     │
     ├── Lint workflow  ──► ruff finds an import error → ❌ you get an email
     ├── Test workflow  ──► pytest passes → ✅
     └── Deploy workflow ──► pushes to HuggingFace → ✅ live demo updated
```

---

## 2. Anatomy of a Workflow File

```yaml
name: My Workflow          # shown in the GitHub Actions UI

on:                        # TRIGGERS — when does this run?
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: "0 8 * * 1"   # every Monday 8am UTC
  workflow_dispatch:        # "Run workflow" button in UI

jobs:                      # a workflow has one or more jobs
  my-job:
    runs-on: ubuntu-latest  # which OS/VM to use

    steps:                  # sequential steps within this job
      - uses: actions/checkout@v4   # a pre-built Action from the marketplace
      - name: My custom step
        run: echo "Hello"           # raw shell command
```

**Key concepts:**
| Term | What it is |
|---|---|
| **Workflow** | A `.yml` file in `.github/workflows/` |
| **Trigger** (`on:`) | What event starts the workflow |
| **Job** | A group of steps that runs on one VM |
| **Step** | One action or shell command inside a job |
| **Runner** | The VM that executes the job (GitHub provides free ones) |
| **Action** | A reusable step from the marketplace (e.g. `actions/checkout`) |
| **Secret** | An encrypted variable stored in GitHub, never visible in logs |
| **Artifact** | A file uploaded from a run for later download |

---

## 3. Workflows in This Repo

### `lint.yml` — runs on every push
The simplest workflow. Uses `ruff` to check Python code quality.

**What to watch in the UI:**
- Open any PR, scroll to the checks section at the bottom
- Click "Details" → see exactly which line number has an issue
- The `--output-format=github` flag makes ruff annotate the diff inline

**Try breaking it:**
```python
# Add this to any .py file and push
import os, sys, json   # ruff will flag unused imports
x = undefined_variable  # ruff will flag undefined name
```

---

### `test.yml` — runs on push to main/develop
Runs pytest for each project in parallel using a **matrix strategy**.

**The matrix strategy** is one of the most powerful Actions features:
```yaml
strategy:
  matrix:
    include:
      - project: 01_realtime_hand_gesture_classifier
      - project: 03_few_shot_image_classifier
```
This creates 2 parallel jobs from one job definition. GitHub shows them
separately in the UI. Total CI time = slowest single project, not sum of all.

**Caching** — notice the `actions/cache` step:
```yaml
key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
```
This fingerprints the requirements file. If it hasn't changed since the last
run, GitHub restores the cached pip packages (~100MB) instead of downloading
from PyPI again. First run: 3 minutes. Subsequent runs with no requirements
change: 20 seconds.

**To add tests for a new project:**
1. Create `your_project/tests/test_something.py`
2. Add the project to the matrix in `test.yml`

---

### `model_validation.yml` — manual or on model file push
An **eval gate** — the model must pass accuracy and sanity checks before
it can be considered deployable.

**Two trigger types combined:**
```yaml
on:
  workflow_dispatch:    # manual button in UI
  push:
    paths:
      - "models/*.pkl"  # only fires when model files change
```

The `paths` filter is crucial for ML repos: you don't want model validation
to run when you fix a README typo.

**`$GITHUB_STEP_SUMMARY`** — a special environment variable pointing to a
file. Anything written to it appears as a formatted Markdown summary on the
workflow run page. Very useful for displaying eval metrics.

**To run manually:**
GitHub → Actions tab → "Model Validation" → "Run workflow" button →
set the minimum accuracy threshold → Run.

---

### `deploy_spaces.yml` — auto-deploy to HuggingFace
Fires when any file in `03_few_shot_image_classifier/` is pushed to main.

**Setup steps (one-time):**
1. Create a HuggingFace Space (huggingface.co → New Space → Gradio)
2. Get your HF token: Settings → Access Tokens → New token (role: write)
3. Add to GitHub: repo Settings → Secrets and variables → Actions → New secret
   - Name: `HF_TOKEN`
   - Value: `hf_xxxxxxxxxx`
4. Edit the workflow: replace `YOUR_USERNAME` with your HF username

**Why use `secrets` instead of putting the token directly?**
Secrets are encrypted and never printed in logs. If you hardcoded
`HF_TOKEN: hf_abc123` in the YAML, it would be public in your repo.

---

### `dependency_audit.yml` — weekly security scan
Runs every Monday automatically via cron schedule.

**Cron syntax** (left to right: minute, hour, day-of-month, month, day-of-week):
```
0  8  *  *  1
│  │  │  │  └── Monday (0=Sunday, 1=Monday, ..., 6=Saturday)
│  │  │  └───── any month
│  │  └──────── any day of month
│  └─────────── 8am
└────────────── 0 minutes past the hour
```

**pip-audit** checks your dependencies against the OSV database
(Open Source Vulnerabilities). If torch 2.0.0 has a CVE, it will flag it.

---

## 4. Setting Up Your Repo

### Step 1: Initialise git
```bash
cd /Users/marcoscosta/Documents/ML_projects
git init
git add .
git commit -m "Initial commit"
```

### Step 2: Create GitHub repo
```bash
# Using GitHub CLI (install with: brew install gh)
gh repo create ml-portfolio --public --source=. --push
```
Or create manually at github.com/new, then:
```bash
git remote add origin https://github.com/YOUR_USERNAME/ml-portfolio.git
git push -u origin main
```

### Step 3: Watch your first workflow run
- GitHub → your repo → Actions tab
- You'll see "Lint" and any other triggered workflows running
- Click on a workflow → click on a job → see each step's output live

### Step 4: Add status badges to your README
After your first run succeeds, add badges to the top of your main README.md:

```markdown
[![Lint](https://github.com/YOUR_USERNAME/ml-portfolio/actions/workflows/lint.yml/badge.svg)](https://github.com/YOUR_USERNAME/ml-portfolio/actions/workflows/lint.yml)
[![Tests](https://github.com/YOUR_USERNAME/ml-portfolio/actions/workflows/test.yml/badge.svg)](https://github.com/YOUR_USERNAME/ml-portfolio/actions/workflows/test.yml)
```

Green badges on your portfolio repo are a strong professional signal.

---

## 5. Common Patterns to Know

### Conditional steps
```yaml
- name: Only runs on main branch
  if: github.ref == 'refs/heads/main'
  run: echo "deploying"

- name: Always upload results, even on failure
  if: always()
  uses: actions/upload-artifact@v4
```

### Passing data between steps
```yaml
- name: Generate a value
  id: my-step          # give the step an ID
  run: echo "value=hello" >> $GITHUB_OUTPUT

- name: Use that value
  run: echo "${{ steps.my-step.outputs.value }}"
```

### Using environment variables
```yaml
env:
  MY_VAR: "hello"        # job-level, available to all steps

steps:
  - name: Step with its own env
    env:
      STEP_VAR: "world"  # step-level only
    run: echo "$MY_VAR $STEP_VAR"
```

### Reusable workflows
You can call one workflow from another:
```yaml
jobs:
  call-lint:
    uses: ./.github/workflows/lint.yml   # call a local workflow
```
Useful when you want deploy to always run lint first.

---

## 6. Free Tier Limits (GitHub Actions)

| Plan | Minutes/month | Storage |
|---|---|---|
| Public repos | **Unlimited** | 500 MB artifacts |
| Private repos (free) | 2,000 min | 500 MB artifacts |

All your portfolio projects are public, so CI is effectively free.
Ubuntu runners cost 1 min/min (1:1). macOS runners cost 10x more — avoid them.

---

## 7. What to Add Next

| Enhancement | Difficulty | Benefit |
|---|---|---|
| Codecov integration (coverage badge) | Easy | Shows test coverage % |
| Pre-commit hooks (run lint locally before push) | Easy | Faster feedback loop |
| Dependabot (auto PRs for dep updates) | Easy | Add `.github/dependabot.yml` |
| Deploy anomaly detection app to Streamlit Cloud | Medium | Another live demo |
| PR comment with model eval results | Medium | Review metrics in PR UI |
| Matrix test across Python 3.10 + 3.11 | Easy | Compatibility guarantee |
