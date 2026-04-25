#!/usr/bin/env bash
set -euo pipefail
# Packages the repository into code.zip excluding virtualenvs and __pycache__
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
ZIP_NAME="relevant-priors-code.zip"
cd "$ROOT_DIR"
echo "Creating $ZIP_NAME..."
zip -r "$ZIP_NAME" . -x "*.venv/*" "*.venv" "./.venv/*" "*/__pycache__/*" "*.pyc" "*.DS_Store"
echo "Created $ZIP_NAME"
