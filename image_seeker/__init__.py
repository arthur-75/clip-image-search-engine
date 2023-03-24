"""Python package searching for the best image."""
from typeguard import install_import_hook

with install_import_hook("image_seeker"):
    from image_seeker.engine import *
