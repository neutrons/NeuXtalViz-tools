MANDI Mesolite Crystal Structure
================================

This tutorial demonstrates how to use the crystal-structure tools in NeuXtalViz
with a Mesolite CIF measured on MANDI.

Workflow
--------

1. Start NeuXtalViz.
2. Open the *Crystal Structure* tool.
3. Use the **Load CIF** button to select the Mesolite CIF used for MANDI.
4. Inspect the lattice parameters and atomic positions on the *Structure* tab.
5. Switch to the *Factors* tab.
6. Enter a suitable :math:`d_{\min}` value and click **Calculate** to compute :math:`F^2`.

Automated example
-----------------

The application script
``tests/applications/crystal_structure.py``
contains a scenario called ``MANDI_Mesolite_Structure`` which:

- Activates the crystal-structure tool.
- Loads the Mesolite CIF from the tests data.
- Shows the *Structure* tab and takes a screenshot.
- Switches to the *Factors* tab and calculates :math:`F^2`.

You can run this scenario (for example under ``xvfb-run``) to regenerate the
screenshots used in the documentation.
