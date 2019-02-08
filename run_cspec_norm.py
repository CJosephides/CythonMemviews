"""
run_cspec_norm.py
"""

import sys
from cspectral_norm import spectral_norm

print("%0.9f" % spectral_norm(int(sys.argv[1])))
