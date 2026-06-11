# Deploy the ACMP demo to Hugging Face Spaces

The demo is a standard Streamlit app (`streamlit_app.py` → `acmp/demo/app.py`).
A Space needs four things at its repo root: this folder's `README.md`
(with the Spaces YAML header), `requirements.txt`, `packages.txt`, plus the
project's `streamlit_app.py` and the `acmp/` package.

## Option A — manual (≈5 minutes)

1. Create a Space: <https://huggingface.co/new-space> → **SDK: Streamlit**, CPU basic (free).
2. Clone it and copy the needed files from this repo:
   ```bash
   git clone https://huggingface.co/spaces/<you>/acmp-demo && cd acmp-demo

   # from a clone of github.com/vc-tr/acmp:
   cp /path/to/acmp/deploy/huggingface/README.md       ./README.md
   cp /path/to/acmp/deploy/huggingface/requirements.txt ./requirements.txt
   cp /path/to/acmp/deploy/huggingface/packages.txt     ./packages.txt
   cp /path/to/acmp/streamlit_app.py                    ./streamlit_app.py
   cp -r /path/to/acmp/acmp                             ./acmp
   cp -r /path/to/acmp/configs                          ./configs
   ```
3. Push: `git add -A && git commit -m "ACMP demo" && git push`.
   The Space builds and serves automatically. The `acmp/` package is importable
   from the repo root, so no `pip install -e .` is needed.

## Option B — auto-sync from GitHub (CI)

Add a workflow that mirrors `main` into the Space on every push. Requires a repo
secret `HF_TOKEN` (a Hugging Face write token) and an existing Space.

```yaml
# .github/workflows/sync-to-hf.yml
name: Sync to HF Spaces
on:
  push:
    branches: [main]
jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - name: Stage Spaces files
        run: |
          cp deploy/huggingface/README.md README.md
          cp deploy/huggingface/requirements.txt requirements.txt
          cp deploy/huggingface/packages.txt packages.txt
          git config user.name github-actions
          git config user.email actions@github.com
          git add -A && git commit -m "spaces build" || true
      - name: Push to Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git push --force \
            https://USER:$HF_TOKEN@huggingface.co/spaces/<you>/acmp-demo \
            HEAD:main
```

## Notes

- **Free-tier scope:** this config installs no `torch`/`diffusers`, so the Space
  renders Ken-Burns motion only. AI animation (Wan VACE) needs a GPU Space.
- Large uploads: Spaces caps request size; keep chapters modest or use a PDF.
- Bump `sdk_version` in `README.md` if HF flags it as unsupported.
