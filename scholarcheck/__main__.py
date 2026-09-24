"""`python -m scholarcheck` runs the same command as the installed `scholarcheck`."""
import sys

from .cli import main

sys.exit(main())
