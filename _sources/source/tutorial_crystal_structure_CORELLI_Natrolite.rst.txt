Crystal Structure with Natrolite Data
====================================

This tutorial demonstrates how to use the crystal-structure tools in
*NeuXtalViz* to work with Natrolite data measured on CORELLI.

Step 1: Load CIF and view structure
-----------------------------------

In the **Crystal Structure** tool:

- Click **Load CIF**.
- Select the Natrolite CIF for the CORELLI dataset.
- The **Structure** tab shows the lattice parameters, space group
  information, unit-cell volume, and a table of atomic positions.
- The 3D view displays the Natrolite framework built from the CIF.

.. figure:: Natrolite_structure_structure.png
   :align: center

   Natrolite crystal structure loaded from CIF (CORELLI).

Step 2: Inspect lattice and scatterers
--------------------------------------

On the **Structure** tab:

- Verify that the lattice constants and angles are reasonable for
  Natrolite.
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

.. figure:: Natrolite_structure_F2.png
   :align: center

   Calculated structure factors :math:`F^2` for Natrolite (CORELLI).
