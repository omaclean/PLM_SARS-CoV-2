"""Self-contained test suite for the PRESCOTT/ESCOTT influenza-diversity pipeline.

This package deliberately sits OUTSIDE ``tests/``.  It has its own ``pytest.ini``
(own testpaths, own markers, own coverage targets) so that this suite and the
repo's pre-existing ``tests/`` suite are completely independent and neither can
break the other.

Run it, from any working directory, with::

    /home3/oml4h/miniconda3/envs/PRESCOTT/bin/python -m pytest \
        /home3/oml4h/PLM_SARS-CoV-2/tests_prescott_iav

The ``__init__.py`` exists so pytest imports the modules as
``tests_prescott_iav.<module>`` (prepend import mode walks up to the first
directory without an ``__init__.py``, i.e. the repository root, and puts *that*
on ``sys.path``).  Two consequences worth knowing:

* test modules may share code via ``from tests_prescott_iav.conftest import ...``;
* module basenames here may safely collide with basenames under ``tests/``.

Nothing in this package may import anything from ``tests/``.
"""

__all__: list[str] = []
