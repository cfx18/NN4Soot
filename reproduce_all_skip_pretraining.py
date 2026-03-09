"""
Run the NN4Soot reproduction pipeline while skipping pretraining.

This is a convenience wrapper around `reproduce_all.py` for the common case
where the pretrained weights already exist and step 01 is too slow.
"""

from reproduce_all import main


if __name__ == "__main__":
    main(default_skip_pretraining=True)
