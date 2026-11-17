import sys
from pathlib import Path

# Let tests `import clipper` without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
