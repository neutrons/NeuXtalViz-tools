Crystal Structure with Scolecite Data
====================================

This tutorial demonstrates how to use the crystal-structure tools in
*NeuXtalViz* to work with Scolecite data measured on TOPAZ.

Step 1: Load CIF and view structure
-----------------------------------

In the **Crystal Structure** tool:

- Click **Load CIF**.
- Select the Scolecite CIF for the TOPAZ dataset.
- The **Structure** tab shows the lattice parameters, space group
  information, unit-cell volume, and a table of atomic positions.
- The 3D view displays the Scolecite framework built from the CIF.

.. figure:: Scolecite_structure_structure.png
   :align: center

   Scolecite crystal structure loaded from CIF (TOPAZ).

Step 2: Inspect lattice and scatterers
--------------------------------------

On the **Structure** tab:

- Verify that the lattice constants and angles are reasonable for
  Scolecite.
- Check the chemical formula, Z value, and unit-cell volume.
- Review the list of scatterers (sites, coordinates, occupancies,
  and U values) to understand the crystal chemistry.

Step 3: Calculate F² for reflections
------------------------------------

Switch to the **Factors** tab:

- Enter a minimum d-spacing in the *d(min)* field.
- Click **Calculate** to compute structure factors and :math:`F^2`
  values.
- Use the resulting table of h, k, l, d, and :math:`F^2` to compare
  against other diffraction software or reference results.

.. figure:: Scolecite_structure_f2.png
   :align: center

   Calculated structure factors :math:`F^2` for Scolecite (TOPAZ).
