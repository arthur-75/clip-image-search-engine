"""Python package searching for the best image."""
from typeguard.importhook import install_import_hook

with install_import_hook("image_seeker"):
    from image_seeker.engine import *