.. _tutorial_topaz_plan:


Experiment Plan with Silicon Data
=================================

This tutorial guides through planning an experiment using the TOPAZ instrument in NeuXtalViz with silicon data.

Step 1: Initialize the Experiment Planner
-----------------------------------------
- Open NeuXtalViz and navigate to the Experiment Planner tab.
- Select "TOPAZ" from the instrument dropdown menu.
- Load the UB matrix and set initial parameters.

.. figure:: Si_plan_initialize.png
   :width: 100%
   :align: center

   Initial experiment planner setup.

Step 2: Update Goniometer Angles
--------------------------------
- Select the goniometer row and set min/max values (e.g., -90 to 90).

.. figure:: Si_plan_update.png
   :width: 100%
   :align: center

   Update goniometer angles.

Step 3: Calculate Peaks
-----------------------
- Enter orientation parameters and calculate desired peaks.

.. figure:: Si_plan_calculate_peak.png
   :width: 100%
   :align: center

   Calculate peaks table.

Step 4: Add Orientation
-----------------------
- Add the calculated orientation to the plan.

.. figure:: Si_plan_add_orientation.png
   :width: 100%
   :align: center

   Add orientation dialog.

Step 5: Optimize Coverage
-------------------------
- Set crystal system, point group, and centering, then optimize coverage.

.. figure:: Si_plan_optimize_coverage.png
   :width: 100%
   :align: center

   Optimize coverage results.

Step 6: Update and Review
-------------------------
- Highlight and update orientations, review info panel for summary.

.. figure:: Si_plan_update_info.png
   :width: 100%
   :align: center

   Info panel for experiment summary.

Step 7: Save and Export
-----------------------
- Save the experiment plan as a CSV file.

.. figure:: Si_plan_save_table_scan.png
   :width: 100%
   :align: center

   Save experiment plan as CSV.
