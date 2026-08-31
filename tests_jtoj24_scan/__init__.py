# Marks tests_jtoj24_scan as a package so `from conftest import ...` resolves the
# same way it does in tests_prescott_iav. The suite is never imported as a
# library; this file exists only for that import symmetry.
