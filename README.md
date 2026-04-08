#
- `src/augment.py`:  main augmentation module
- `augment.py`: simple root-level runner
- `clear.py`: clears generated augmentation outputs
- `raws/`, `upsampled/`, `augmented/`: data directories you can keep using

## Current augment features

- temporally consistent exposure variations
- deterministic coarse occlusion / blocking
- small translation / rotation / horizontal flip perturbations
- batch processing over `.mp4` and `.mov` clips in a directory

## Run it

From the repo root:

```bash
python3 augment.py
```

Common examples:

```bash
python3 augment.py 3 -all
python3 augment.py 5 -exposure
python3 augment.py 2 -blocking -translation
python3 augment.py 3 -all -upsampled
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

Installed console script:

```bash
tackle-vision-ml
```

## Fresh-start notes

- `assign_label()` in `src/tackle_vision_augment.py` is a placeholder for your own labeling logic.
- The augmentation module is intentionally self-contained so you can refactor it into a larger architecture later.
- There are no pipeline tests left; this is meant to be a reset point, not a polished package.
