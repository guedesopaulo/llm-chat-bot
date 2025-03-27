# Manual Installation
```bash
# 1. Create and activate the Conda environment
conda env create -n fav-ver -f environment.yml
conda activate fav-ver

# 3. Lock base dependencies from setup.cfg into requirements.txt
uv pip compile setup.cfg -o requirements.txt

# 5. Sync dependencies exactly as locked into your environment
uv pip sync requirements.txt
```

Everytime dependencies in setup.cfg changes, run again

```
uv pip compile setup.cfg -o requirements.txt

uv pip compile setup.cfg -o requirements-dev.txt --extra dev
```