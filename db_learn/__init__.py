# Authors: Pierre Laforgue <pierre.laforgue@telecom-paristech.fr>
#
# License: MIT
"""Debiased ERM in Python"""

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.devN' where N is an integer.
#

__version__ = '0.1.dev0'

from .db_weights import (compute_weights, compute_Omegas, compute_Ws, mk_Momega,
                         one_sample)  # noqa
from .db_funcs import (norm_in_bnd_vec, norm_out_bnd_vec, dim_in_bnd_vec,
                       dim_out_bnd_vec, dim_in_set_vec, Gauss, SampleX)  # noqa
