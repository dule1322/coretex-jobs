#!/bin/bash

eval "$(conda shell.bash hook)"
dir=$1
echo "Checking directory: $dir"

# Skip the directory if no .mypy.ini file is found
if [ ! -f "$dir/.mypy.ini" ]; then
  echo "No .mypy.ini file found in $dir, skipping..."
  exit 0
fi

if [ -f "$dir/environment.yml" ]; then
  echo "Setting up conda environment for $dir"
  conda env create -n $(basename "$dir") -f "$dir/environment.yml"
  echo "Created conda environment"
  conda activate $(basename "$dir")
  pip install mypy
elif [ -f "$dir/requirements.txt" ]; then
  echo "Setting up venv for $dir"
  python3.9 -m venv "$dir/venv"
  echo "activate venv"
  source "$dir/venv/bin/activate"
  echo "install requirements"
  pip install --upgrade pip
  pip install -r "$dir/requirements.txt"
  pip install mypy
fi

echo "Running mypy in $dir"
set +e  # Disable exit on error
mypy_output=$(mypy --config-file "$dir/.mypy.ini" "$dir" 2>&1)
set -e  # Re-enable exit on error

echo "$mypy_output"
if echo "$mypy_output" | grep -q 'error:'; then
  echo "Running install-types in $dir"
  mypy --install-types --non-interactive --config-file "$dir/.mypy.ini" "$dir"
fi

if [ -f "$dir/environment.yml" ]; then
  conda deactivate
  conda remove -y -n $(basename "$dir") --all
elif [ -f "$dir/requirements.txt" ]; then
  deactivate
  rm -rf "$dir/venv"
fi

# done
