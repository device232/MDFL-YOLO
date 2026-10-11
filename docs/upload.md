# Publish this code on GitHub

The directory contains only source code, configuration examples, documentation, and tests. Data, trained weights, manuscript files, environment files, and generated outputs are excluded by `.gitignore`.

## Web upload

1. Sign in to GitHub and create a public repository named `MDFL-YOLO`.
2. Upload the contents of this directory, keeping the folder structure.
3. Use the resulting repository address in the manuscript's Code availability statement.
4. Create a release tag after checking the uploaded version, so the manuscript can identify a fixed version.

The `.github/workflows/tests.yml` file enables CPU correctness checks on pushes and pull requests.

## Command-line upload

Run these commands from this directory after installing and signing in with GitHub CLI. Replace `YOUR_ACCOUNT` with the destination account. Git must already have your author identity configured.

```bash
gh auth login
git add .
git commit -m "Release MDFL-YOLO implementation"
gh repo create YOUR_ACCOUNT/MDFL-YOLO --public --source . --remote origin --push
```

For a repository created through the website, set its address as `origin` and push the `main` branch instead:

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/MDFL-YOLO.git
git push -u origin main
```
