Tutorial: UB Tools with TOPAZ Silicon Data
=========================================

This tutorial demonstrates how to use the UB tools in *NeuXtalViz* to process TOPAZ Silicon data, following the steps in the automated test script.

.. figure:: Si_UB_convert_Q.png
   :width: 100%
   :align: center

   Convert to Q workspace.

Step 1: Select Instrument and Data
----------------------------------
- Set the instrument to **TOPAZ**.
- Enter IPTS number: ``36169``.
- Enter run number: ``54145``.
- Click **Convert to Q** to load and convert the data.

.. figure:: Si_UB_find_peaks.png
   :width: 100%
   :align: center

   Find Peaks.

Step 2: Find Peaks
------------------
- Adjust peak finding parameters as needed:
  - Max peaks
  - Density threshold
  - Min distance
  - Click **Find Peaks** to identify candidate peaks.

.. figure:: Si_UB_primitive_cell.png
   :width: 100%
   :align: center

   Primitive Cell Calculation.

Step 3: Primitive Cell Calculation
----------------------------------
- Set tolerance and constraints for cell finding.
- Click **Niggli** to calculate the primitive cell.

.. figure:: Si_UB_conventional_cell.png
   :width: 100%
   :align: center

   Select Conventional Cell.

Step 4: Select Conventional Cell
-------------------------------
- Select the desired cell from the table.
- Click **Select** to set the conventional cell.

.. figure:: ub_tab_reSi_UB_refine_UBfine_UB.png
   :width: 100%
   :align: center

   Refine UB Matrix.

Step 5: Refine UB Matrix
------------------------
- Switch to the UB tab.
- Set the optimization method (e.g., **Cubic**).
- Click **Refine UB** to optimize the UB matrix.

This workflow matches the automated test and can be used as a step-by-step guide for new users.
