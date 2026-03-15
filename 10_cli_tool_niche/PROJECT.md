# CLI Tool for Something Niche

## Overview
A fast, focused Python or Rust CLI that solves a specific developer pain point. Options: bulk PDF metadata stripper, image EXIF anonymizer, or a local-first LLM prompt templating tool.

## Category
Software Engineering / Developer Tooling

## Candidate Ideas (pick one)

### Option A: Image EXIF Anonymizer (`exif-clean`)
Strip GPS coordinates, device info, timestamps, and author metadata from images in bulk.
- `exif-clean ./photos/ --recursive --output ./clean/`
- Shows: file I/O, metadata parsing, privacy tooling

### Option B: PDF Metadata Stripper (`pdf-scrub`)
Remove author, title, creation date, software metadata from PDFs.
- `pdf-scrub report.pdf --output clean_report.pdf`
- Pairs well with MedAnon angle

### Option C: Local LLM Prompt Templating Tool (`prompter`)
A CLI for managing and rendering prompt templates with variable substitution, stored locally as YAML/Markdown files.
- `prompter run summarize --var text="$(cat article.txt)"`
- Shows: modern LLM tooling awareness, developer ergonomics

## Stack
- **Python** with `click` or `typer` for CLI
- **Rust** with `clap` if performance is the story
- **PyPI / crates.io** for distribution
- **GitHub Actions** — automated releases

## Key Features (all options)
- Fast, zero-config default behavior
- Clear `--help` output
- `--dry-run` flag
- Verbose logging option
- Installable via `pip install` or `cargo install`

## Portfolio Value
- Shows software craft: clean API design, good UX
- Publishable to PyPI/crates.io — a live artifact
- Niche tools get GitHub stars from practitioners

## Milestones
- [ ] Pick one option
- [ ] Core functionality
- [ ] CLI interface with help text
- [ ] Tests
- [ ] PyPI / crates.io packaging
- [ ] GitHub release with binaries (if Rust)
- [ ] README with usage examples

## Notes
<!-- Add implementation notes, decisions, and progress here -->
