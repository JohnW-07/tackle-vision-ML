from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    target_dirs = [project_root / "augmented", project_root / "finals"]

    for target_dir in target_dirs:
        if not target_dir.exists():
            print(f"Directory not found: {target_dir}")
            continue

        deleted = 0
        for path in target_dir.rglob("*"):
            if path.is_file():
                path.unlink()
                deleted += 1

        print(f"Deleted {deleted} file(s) from {target_dir}")


if __name__ == "__main__":
    main()
