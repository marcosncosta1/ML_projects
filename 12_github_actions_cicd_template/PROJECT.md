# GitHub Actions CI/CD Template for ML Projects

## Overview
A reusable GitHub Actions workflow template repository for ML projects, covering the full lifecycle: lint → test → train → evaluate → deploy. Shows DevOps maturity for ML.

## Category
Software Engineering / DevOps / MLOps

## Stack
- **GitHub Actions** — CI/CD orchestration
- **Python + pytest** — testing
- **ruff / black / isort** — linting and formatting
- **DVC or MLflow** — experiment tracking (optional)
- **Docker** — containerized training/serving
- **HuggingFace Spaces / Streamlit Cloud** — auto-deploy target

## Workflow Stages

### 1. Lint
```yaml
- ruff check .
- black --check .
- isort --check .
```

### 2. Test
```yaml
- pytest tests/ --cov=src --cov-report=xml
- Upload coverage to Codecov
```

### 3. Train (on tag or manual trigger)
```yaml
- Run training script
- Log metrics to MLflow / W&B
- Save model artifact
```

### 4. Evaluate
```yaml
- Run evaluation script
- Post metrics as PR comment
- Fail if metrics drop below threshold
```

### 5. Deploy (on main merge)
```yaml
- Build Docker image
- Push to registry
- Trigger HuggingFace Spaces or Streamlit Cloud redeploy
```

## Repository Structure
```
ml-cicd-template/
├── .github/
│   └── workflows/
│       ├── ci.yml          # lint + test on every PR
│       ├── train.yml       # train on tag or manual trigger
│       └── deploy.yml      # deploy on main merge
├── template/               # copy-paste starter for new ML projects
│   ├── src/
│   ├── tests/
│   └── Dockerfile
├── docs/
│   └── how-to-use.md
└── README.md
```

## Portfolio Value
- Highly sought-after MLOps skill
- Reusable = genuine utility, not just a demo
- Shows software engineering maturity beyond model training
- Can be used as a template for all other portfolio projects

## Milestones
- [ ] `ci.yml` — lint + test workflow
- [ ] `train.yml` — training + artifact upload
- [ ] `evaluate.yml` — metrics evaluation + PR comment
- [ ] `deploy.yml` — HuggingFace Spaces deploy
- [ ] Starter template directory
- [ ] README with usage guide + badges
- [ ] Publish as GitHub Template Repository

## Notes
<!-- Add implementation notes, decisions, and progress here -->
