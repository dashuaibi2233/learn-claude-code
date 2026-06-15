from pathlib import Path

basedir = Path.cwd()

file_path = (basedir / "test.txt").resolve()

print(file_path)

print(file_path.is_relative_to(basedir))