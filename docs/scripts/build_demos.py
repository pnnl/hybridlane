import re
import subprocess
from pathlib import Path

docs_dir = Path(__file__).parent.parent.absolute()
source_dir = docs_dir.parent / "examples" / "demos"
target_dir = docs_dir / "source" / "demos"

if __name__ == "__main__":
    target_dir.mkdir(parents=True, exist_ok=True)

    # Find all marimo notebooks in the root directory of examples/demos and compile them
    # to markdown in the target directory

    for notebook in source_dir.glob("*.py"):
        md_filename = notebook.stem + ".md"
        md_file = target_dir / md_filename

        if md_file.exists():
            print(f"Skipping {notebook.name} because {md_file.name} already exists.")
            continue
        else:
            print(f"Rendering notebook {notebook.name}")

        res = subprocess.run(
            [
                "uv",
                "run",
                "--with",
                "marimo-md-export",
                "marimo-md-export",
                str(notebook),
                str(target_dir / md_filename),
            ],
            capture_output=True,
        )

        if res.returncode != 0:
            print(f"Error rendering {notebook.name}: {res.stderr.decode()}")
            exit(1)

        # Strip leading "NN " number prefix from the frontmatter title,
        # and inject tocdepth: 1 so only the page title appears in the nav sidebar
        content = md_file.read_text()
        content = re.sub(r"^(title:\s*)\d+\s+", r"\1", content, flags=re.MULTILINE)
        content = re.sub(r"^(title:.+)$", r"\1\ntocdepth: 1", content, flags=re.MULTILINE)
        md_file.write_text(content)
