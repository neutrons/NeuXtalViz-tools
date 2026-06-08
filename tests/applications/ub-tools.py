import os
import sys
import subprocess

from qtpy.QtTest import QTest
from qtpy.QtCore import Qt

from utilities import run_qt_scenario, copy_generated_pngs

DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def TOPAZ_Si_UB(app, window):
    directory = os.path.join(DIRECTORY, "TOPAZ")

    app_stack = window.centralWidget().layout().itemAt(0).widget()
    app_stack.setCurrentIndex(0)

    ub_presenter = window.ub
    ub_view = ub_presenter.view

    index = ub_view.instrument_combo.findText("TOPAZ")
    ub_view.instrument_combo.setCurrentIndex(index)
    ub_view.ipts_line.setText("36169")
    ub_view.runs_line.setText("54145")
    ub_view.wl_min_line.setText("0.4")
    ub_view.wl_max_line.setText("3.5")

    ub_view.instrument_combo.setStyleSheet("background-color: yellow;")
    ub_view.ipts_line.setStyleSheet("background-color: yellow;")
    ub_view.runs_line.setStyleSheet("background-color: yellow;")
    ub_view.convert_to_q_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.convert_to_q_button, Qt.LeftButton)
    QTest.qWait(1000 * 80)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_UB_convert_Q.png"), "png"
    )

    ub_view.instrument_combo.setStyleSheet("")
    ub_view.ipts_line.setStyleSheet("")
    ub_view.runs_line.setStyleSheet("")
    ub_view.convert_to_q_button.setStyleSheet("")

    ub_view.max_peaks_line.setStyleSheet("background-color: yellow;")
    ub_view.density_threshold_line.setStyleSheet("background-color: yellow;")
    ub_view.min_distance_line.setStyleSheet("background-color: yellow;")
    ub_view.find_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.find_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_UB_find_peaks.png"), "png"
    )

    ub_view.max_peaks_line.setStyleSheet("")
    ub_view.density_threshold_line.setStyleSheet("")
    ub_view.min_distance_line.setStyleSheet("")
    ub_view.find_button.setStyleSheet("")

    ub_view.calculate_tolerance_line.setStyleSheet("background-color: yellow;")
    ub_view.min_const_line.setStyleSheet("background-color: yellow;")
    ub_view.max_const_line.setStyleSheet("background-color: yellow;")
    ub_view.niggli_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.niggli_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_UB_primitive_cell.png"), "png"
    )

    ub_view.calculate_tolerance_line.setStyleSheet("")
    ub_view.min_const_line.setStyleSheet("")
    ub_view.max_const_line.setStyleSheet("")
    ub_view.niggli_button.setStyleSheet("")

    ub_view.cell_table.selectRow(0)
    ub_view.select_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.select_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_UB_conventional_cell.png"), "png"
    )

    ub_view.cell_table.clearSelection()
    ub_view.select_button.setStyleSheet("")

    ub_view.ub_tab.setCurrentIndex(2)

    index = ub_view.optimize_combo.findText("Cubic")
    ub_view.optimize_combo.setCurrentIndex(index)

    ub_view.optimize_combo.setStyleSheet("background-color: yellow;")
    ub_view.refine_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.refine_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_UB_refine_UB.png"), "png"
    )

    ub_view.optimize_combo.setStyleSheet("")
    ub_view.refine_button.setStyleSheet("")

    ub_view.tab_widget.setCurrentIndex(1)
    ub_view.peaks_table.selectRow(0)

    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_UB_view_peak.png"), "png"
    )

    ub_view.peaks_table.clearSelection()

    ub_view.tab_widget.setCurrentIndex(2)

    ub_view.slice_combo.setStyleSheet("background-color: yellow;")
    ub_view.slice_line.setStyleSheet("background-color: yellow;")
    ub_view.convert_to_hkl_button.setStyleSheet("background-color: yellow;")

    QTest.mouseClick(ub_view.convert_to_hkl_button, Qt.LeftButton)
    QTest.qWait(1000 * 15)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_UB_slice_view.png"), "png"
    )

    ub_view.slice_combo.setStyleSheet("")
    ub_view.slice_line.setStyleSheet("")
    ub_view.convert_to_hkl_button.setStyleSheet("")

    ub_view.tab_widget.setCurrentIndex(0)

    ub_view.save_ub_button.setStyleSheet("background-color: yellow;")

    QTest.qWait(1000 * 1)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_UB_save_UB.png"), "png"
    )
    ub_presenter.model.save_UB(
        os.path.join("/SNS/TOPAZ/IPTS-36169/shared/nxv/", "Si_UB.mat")
    )

    copy_generated_pngs(directory)


def TOPAZ_Scolecite_UB(app, window):
    directory = os.path.join(DIRECTORY, "TOPAZ")

    app_stack = window.centralWidget().layout().itemAt(0).widget()
    app_stack.setCurrentIndex(0)

    ub_presenter = window.ub
    ub_view = ub_presenter.view

    index = ub_view.instrument_combo.findText("TOPAZ")
    ub_view.instrument_combo.setCurrentIndex(index)
    ub_view.ipts_line.setText("31856")
    ub_view.runs_line.setText("50024")
    ub_view.wl_min_line.setText("0.4")
    ub_view.wl_max_line.setText("3.5")

    cal_path = "/SNS/TOPAZ/shared/calibration/2024B/TOPAZ_2024B_AG.DetCal"
    ub_view.cal_line.setText(cal_path)

    ub_view.instrument_combo.setStyleSheet("background-color: yellow;")
    ub_view.ipts_line.setStyleSheet("background-color: yellow;")
    ub_view.runs_line.setStyleSheet("background-color: yellow;")
    ub_view.cal_line.setStyleSheet("background-color: yellow;")
    ub_view.convert_to_q_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.convert_to_q_button, Qt.LeftButton)
    QTest.qWait(1000 * 80)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Scolecite_UB_convert_Q.png"), "png"
    )

    ub_view.instrument_combo.setStyleSheet("")
    ub_view.ipts_line.setStyleSheet("")
    ub_view.runs_line.setStyleSheet("")
    ub_view.cal_line.setStyleSheet("")
    ub_view.convert_to_q_button.setStyleSheet("")

    ub_view.max_peaks_line.setStyleSheet("background-color: yellow;")
    ub_view.density_threshold_line.setStyleSheet("background-color: yellow;")
    ub_view.min_distance_line.setStyleSheet("background-color: yellow;")
    ub_view.find_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.find_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Scolecite_UB_find_peaks.png"), "png"
    )

    ub_view.max_peaks_line.setStyleSheet("")
    ub_view.density_threshold_line.setStyleSheet("")
    ub_view.min_distance_line.setStyleSheet("")
    ub_view.find_button.setStyleSheet("")

    ub_view.calculate_tolerance_line.setStyleSheet("background-color: yellow;")
    ub_view.min_const_line.setStyleSheet("background-color: yellow;")
    ub_view.max_const_line.setStyleSheet("background-color: yellow;")
    ub_view.niggli_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.niggli_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Scolecite_UB_primitive_cell.png"), "png"
    )

    ub_view.calculate_tolerance_line.setStyleSheet("")
    ub_view.min_const_line.setStyleSheet("")
    ub_view.max_const_line.setStyleSheet("")
    ub_view.niggli_button.setStyleSheet("")

    ub_view.cell_table.selectRow(1)
    ub_view.select_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.select_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Scolecite_UB_conventional_cell.png"), "png"
    )

    ub_view.cell_table.clearSelection()
    ub_view.select_button.setStyleSheet("")

    ub_view.ub_tab.setCurrentIndex(2)

    index = ub_view.optimize_combo.findText("Monoclinic")
    ub_view.optimize_combo.setCurrentIndex(index)

    ub_view.optimize_combo.setStyleSheet("background-color: yellow;")
    ub_view.refine_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.refine_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Scolecite_UB_refine_UB.png"), "png"
    )

    ub_view.optimize_combo.setStyleSheet("")
    ub_view.refine_button.setStyleSheet("")

    ub_view.tab_widget.setCurrentIndex(1)
    ub_view.peaks_table.selectRow(0)

    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Scolecite_UB_view_peak.png"), "png"
    )

    ub_view.peaks_table.clearSelection()

    ub_view.tab_widget.setCurrentIndex(2)

    ub_view.slice_combo.setStyleSheet("background-color: yellow;")
    ub_view.slice_line.setStyleSheet("background-color: yellow;")
    ub_view.convert_to_hkl_button.setStyleSheet("background-color: yellow;")

    QTest.mouseClick(ub_view.convert_to_hkl_button, Qt.LeftButton)
    QTest.qWait(1000 * 15)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Scolecite_UB_slice_view.png"), "png"
    )

    ub_view.slice_combo.setStyleSheet("")
    ub_view.slice_line.setStyleSheet("")
    ub_view.convert_to_hkl_button.setStyleSheet("")

    ub_view.tab_widget.setCurrentIndex(0)

    ub_view.save_ub_button.setStyleSheet("background-color: yellow;")

    QTest.qWait(1000 * 1)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Scolecite_UB_save_UB.png"), "png"
    )
    ub_presenter.model.save_UB(
        os.path.join("/SNS/TOPAZ/IPTS-31856/shared/nxv/", "Scolecite_UB.mat")
    )

    copy_generated_pngs(directory)


def CORELLI_Bixbyite_UB(app, window):
    directory = os.path.join(DIRECTORY, "CORELLI")

    app_stack = window.centralWidget().layout().itemAt(0).widget()
    app_stack.setCurrentIndex(0)

    ub_presenter = window.ub
    ub_view = ub_presenter.view

    index = ub_view.instrument_combo.findText("CORELLI")
    ub_view.instrument_combo.setCurrentIndex(index)
    ub_view.ipts_line.setText("36170")
    ub_view.runs_line.setText("37055")
    ub_view.wl_min_line.setText("0.6")
    ub_view.wl_max_line.setText("2.5")

    ub_view.instrument_combo.setStyleSheet("background-color: yellow;")
    ub_view.ipts_line.setStyleSheet("background-color: yellow;")
    ub_view.runs_line.setStyleSheet("background-color: yellow;")
    ub_view.convert_to_q_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.convert_to_q_button, Qt.LeftButton)
    QTest.qWait(1000 * 120)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_UB_convert_Q.png"), "png"
    )

    ub_view.instrument_combo.setStyleSheet("")
    ub_view.ipts_line.setStyleSheet("")
    ub_view.runs_line.setStyleSheet("")
    ub_view.convert_to_q_button.setStyleSheet("")

    ub_view.min_distance_line.setText("0.1")

    ub_view.max_peaks_line.setStyleSheet("background-color: yellow;")
    ub_view.density_threshold_line.setStyleSheet("background-color: yellow;")
    ub_view.min_distance_line.setStyleSheet("background-color: yellow;")
    ub_view.find_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.find_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_UB_find_peaks.png"), "png"
    )

    ub_view.max_peaks_line.setStyleSheet("")
    ub_view.density_threshold_line.setStyleSheet("")
    ub_view.min_distance_line.setStyleSheet("")
    ub_view.find_button.setStyleSheet("")

    ub_view.calculate_tolerance_line.setStyleSheet("background-color: yellow;")
    ub_view.min_const_line.setStyleSheet("background-color: yellow;")
    ub_view.max_const_line.setStyleSheet("background-color: yellow;")
    ub_view.niggli_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.niggli_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_UB_primitive_cell.png"), "png"
    )

    ub_view.calculate_tolerance_line.setStyleSheet("")
    ub_view.min_const_line.setStyleSheet("")
    ub_view.max_const_line.setStyleSheet("")
    ub_view.niggli_button.setStyleSheet("")

    ub_view.cell_table.selectRow(0)
    ub_view.select_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.select_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_UB_conventional_cell.png"), "png"
    )

    ub_view.cell_table.clearSelection()
    ub_view.select_button.setStyleSheet("")

    ub_view.ub_tab.setCurrentIndex(1)

    ub_view.T11_line.setText("-1")
    ub_view.T22_line.setText("0")
    ub_view.T23_line.setText("1")
    ub_view.T32_line.setText("1")
    ub_view.T33_line.setText("0")

    ub_view.T11_line.setStyleSheet("background-color: yellow;")
    ub_view.T22_line.setStyleSheet("background-color: yellow;")
    ub_view.T23_line.setStyleSheet("background-color: yellow;")
    ub_view.T32_line.setStyleSheet("background-color: yellow;")
    ub_view.T33_line.setStyleSheet("background-color: yellow;")
    ub_view.transform_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.transform_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_UB_transform_cell.png"), "png"
    )

    ub_view.T11_line.setStyleSheet("")
    ub_view.T22_line.setStyleSheet("")
    ub_view.T23_line.setStyleSheet("")
    ub_view.T32_line.setStyleSheet("")
    ub_view.T33_line.setStyleSheet("")
    ub_view.transform_button.setStyleSheet("")

    ub_view.ub_tab.setCurrentIndex(2)

    index = ub_view.optimize_combo.findText("Cubic")
    ub_view.optimize_combo.setCurrentIndex(index)

    ub_view.optimize_combo.setStyleSheet("background-color: yellow;")
    ub_view.refine_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.refine_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_UB_refine_UB.png"), "png"
    )

    ub_view.optimize_combo.setStyleSheet("")
    ub_view.refine_button.setStyleSheet("")

    ub_view.tab_widget.setCurrentIndex(1)
    ub_view.peaks_table.selectRow(0)

    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_UB_view_peak.png"), "png"
    )

    ub_view.peaks_table.clearSelection()
    ub_view.tab_widget.setCurrentIndex(2)

    ub_view.slice_combo.setStyleSheet("background-color: yellow;")
    ub_view.slice_line.setStyleSheet("background-color: yellow;")
    ub_view.convert_to_hkl_button.setStyleSheet("background-color: yellow;")

    QTest.mouseClick(ub_view.convert_to_hkl_button, Qt.LeftButton)
    QTest.qWait(1000 * 15)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_UB_slice_view.png"), "png"
    )

    ub_view.slice_combo.setStyleSheet("")
    ub_view.slice_line.setStyleSheet("")
    ub_view.convert_to_hkl_button.setStyleSheet("")

    ub_view.tab_widget.setCurrentIndex(0)

    ub_view.save_ub_button.setStyleSheet("background-color: yellow;")

    QTest.qWait(1000 * 1)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_UB_save_UB.png"), "png"
    )
    ub_presenter.model.save_UB(
        os.path.join("/SNS/CORELLI/IPTS-36170/shared/nxv/", "Bixbyite_UB.mat")
    )

    copy_generated_pngs(directory)


def MANDI_Mesolite_UB(app, window):
    directory = os.path.join(DIRECTORY, "MANDI")

    app_stack = window.centralWidget().layout().itemAt(0).widget()
    app_stack.setCurrentIndex(0)

    ub_presenter = window.ub
    ub_view = ub_presenter.view

    index = ub_view.instrument_combo.findText("MANDI")
    ub_view.instrument_combo.setCurrentIndex(index)
    ub_view.ipts_line.setText("8776")
    ub_view.runs_line.setText("11612")
    ub_view.convert_min_d_line.setText("1.0")
    ub_view.wl_min_line.setText("2")
    ub_view.wl_max_line.setText("4")

    cal_path = "/SNS/MANDI/shared/calibration/2024B/calibration.DetCal"
    ub_view.cal_line.setText(cal_path)

    ub_view.instrument_combo.setStyleSheet("background-color: yellow;")
    ub_view.ipts_line.setStyleSheet("background-color: yellow;")
    ub_view.runs_line.setStyleSheet("background-color: yellow;")
    ub_view.convert_min_d_line.setStyleSheet("background-color: yellow;")
    ub_view.cal_line.setStyleSheet("background-color: yellow;")
    ub_view.convert_to_q_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.convert_to_q_button, Qt.LeftButton)
    QTest.qWait(1000 * 80)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Mesolite_UB_convert_Q.png"), "png"
    )

    ub_view.instrument_combo.setStyleSheet("")
    ub_view.ipts_line.setStyleSheet("")
    ub_view.runs_line.setStyleSheet("")
    ub_view.convert_min_d_line.setStyleSheet("")
    ub_view.cal_line.setStyleSheet("")
    ub_view.convert_to_q_button.setStyleSheet("")

    ub_view.max_peaks_line.setStyleSheet("background-color: yellow;")
    ub_view.density_threshold_line.setStyleSheet("background-color: yellow;")
    ub_view.min_distance_line.setStyleSheet("background-color: yellow;")
    ub_view.find_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.find_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Mesolite_UB_find_peaks.png"), "png"
    )

    ub_view.max_peaks_line.setStyleSheet("")
    ub_view.density_threshold_line.setStyleSheet("")
    ub_view.min_distance_line.setStyleSheet("")
    ub_view.find_button.setStyleSheet("")

    ub_view.a_line.setText("18.4")
    ub_view.b_line.setText("56.6")
    ub_view.c_line.setText("6.54")
    ub_view.alpha_line.setText("90")
    ub_view.beta_line.setText("90")
    ub_view.gamma_line.setText("90")
    ub_view.calculate_tolerance_line.setText("0.15")

    ub_view.a_line.setStyleSheet("background-color: yellow;")
    ub_view.b_line.setStyleSheet("background-color: yellow;")
    ub_view.c_line.setStyleSheet("background-color: yellow;")
    ub_view.alpha_line.setStyleSheet("background-color: yellow;")
    ub_view.beta_line.setStyleSheet("background-color: yellow;")
    ub_view.gamma_line.setStyleSheet("background-color: yellow;")
    ub_view.calculate_tolerance_line.setStyleSheet("background-color: yellow;")

    QTest.mouseClick(ub_view.conventional_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    ub_view.a_line.setStyleSheet("")
    ub_view.b_line.setStyleSheet("")
    ub_view.c_line.setStyleSheet("")
    ub_view.alpha_line.setStyleSheet("")
    ub_view.beta_line.setStyleSheet("")
    ub_view.gamma_line.setStyleSheet("")
    ub_view.calculate_tolerance_line.setStyleSheet("")
    ub_view.optimize_combo.setStyleSheet("")
    ub_view.conventional_button.setStyleSheet("")
    ub_view.refine_button.setStyleSheet("")

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Mesolite_UB_conventional_cell.png"), "png"
    )

    ub_view.index_tolerance_line.setText("0.15")

    ub_view.peaks_tab.setCurrentIndex(1)
    ub_view.index_button.setStyleSheet("background-color: green;")
    ub_view.index_tolerance_line.setStyleSheet("background-color: yellow;")

    QTest.mouseClick(ub_view.index_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Mesolite_UB_index_peaks.png"), "png"
    )

    ub_view.index_button.setStyleSheet("")
    ub_view.index_tolerance_line.setStyleSheet("")

    ub_view.ub_tab.setCurrentIndex(2)

    index = ub_view.optimize_combo.findText("Orthorhombic")
    ub_view.optimize_combo.setCurrentIndex(index)
    ub_view.refine_tolerance_line.setText("0.15")

    ub_view.optimize_combo.setStyleSheet("background-color: yellow;")
    ub_view.refine_tolerance_line.setStyleSheet("background-color: yellow;")
    ub_view.conventional_button.setStyleSheet("background-color: green;")

    ub_view.refine_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.refine_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Mesolite_UB_refine_UB.png"), "png"
    )

    ub_view.optimize_combo.setStyleSheet("")
    ub_view.refine_tolerance_line.setStyleSheet("")
    ub_view.conventional_button.setStyleSheet("")

    ub_view.refine_button.setStyleSheet("")

    ub_view.tab_widget.setCurrentIndex(1)
    ub_view.peaks_table.selectRow(0)

    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Mesolite_UB_view_peak.png"), "png"
    )

    ub_view.peaks_table.clearSelection()

    ub_view.tab_widget.setCurrentIndex(2)

    ub_view.slice_combo.setStyleSheet("background-color: yellow;")
    ub_view.slice_line.setStyleSheet("background-color: yellow;")
    ub_view.convert_to_hkl_button.setStyleSheet("background-color: yellow;")

    QTest.mouseClick(ub_view.convert_to_hkl_button, Qt.LeftButton)
    QTest.qWait(1000 * 15)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Mesolite_UB_slice_view.png"), "png"
    )

    ub_view.slice_combo.setStyleSheet("")
    ub_view.slice_line.setStyleSheet("")
    ub_view.convert_to_hkl_button.setStyleSheet("")

    ub_view.tab_widget.setCurrentIndex(0)

    ub_view.save_ub_button.setStyleSheet("background-color: yellow;")

    QTest.qWait(1000 * 1)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Mesolite_UB_save_UB.png"), "png"
    )
    ub_presenter.model.save_UB(
        os.path.join("/SNS/MANDI/IPTS-8776/shared/nxv/", "Mesolite_UB.mat")
    )

    copy_generated_pngs(directory)


def CORELLI_Natrolite_UB(app, window):
    directory = os.path.join(DIRECTORY, "CORELLI")

    app_stack = window.centralWidget().layout().itemAt(0).widget()
    app_stack.setCurrentIndex(0)

    ub_presenter = window.ub
    ub_view = ub_presenter.view

    index = ub_view.instrument_combo.findText("CORELLI")
    ub_view.instrument_combo.setCurrentIndex(index)
    ub_view.ipts_line.setText("31429")
    ub_view.runs_line.setText("383673")
    ub_view.wl_min_line.setText("0.6")
    ub_view.wl_max_line.setText("2.5")

    cal_path = "/SNS/CORELLI/shared/calibration/2022A/calibration.xml"
    ub_view.cal_line.setText(cal_path)

    ub_view.instrument_combo.setStyleSheet("background-color: yellow;")
    ub_view.ipts_line.setStyleSheet("background-color: yellow;")
    ub_view.runs_line.setStyleSheet("background-color: yellow;")
    ub_view.cal_line.setStyleSheet("background-color: yellow;")
    ub_view.convert_to_q_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.convert_to_q_button, Qt.LeftButton)
    QTest.qWait(1000 * 80)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Natrolite_UB_convert_Q.png"), "png"
    )

    ub_view.instrument_combo.setStyleSheet("")
    ub_view.ipts_line.setStyleSheet("")
    ub_view.runs_line.setStyleSheet("")
    ub_view.cal_line.setStyleSheet("")
    ub_view.convert_to_q_button.setStyleSheet("")

    ub_view.tab_widget.setCurrentIndex(2)
    ub_view.inspect_verify_tab.setCurrentIndex(1)

    ub_view.horizontal_line.setStyleSheet("background-color: yellow;")
    ub_view.vertical_line.setStyleSheet("background-color: yellow;")
    ub_view.diffraction_line.setStyleSheet("background-color: yellow;")
    ub_view.add_peak_button.setStyleSheet("background-color: green;")

    ub_view.horizontal_line.setText("-11")
    ub_view.vertical_line.setText("-12")
    ub_view.diffraction_line.setText("1.83")
    ub_presenter.update_roi()
    ub_presenter.update_scan()

    QTest.mouseClick(ub_view.add_peak_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Natrolite_UB_first_peak.png"), "png"
    )

    ub_view.horizontal_line.setText("31")
    ub_view.vertical_line.setText("-6")
    ub_view.diffraction_line.setText("1.73")
    ub_presenter.update_roi()
    ub_presenter.update_scan()

    QTest.mouseClick(ub_view.add_peak_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Natrolite_UB_second_peak.png"), "png"
    )

    ub_view.horizontal_line.setStyleSheet("")
    ub_view.vertical_line.setStyleSheet("")
    ub_view.diffraction_line.setStyleSheet("")
    ub_view.add_peak_button.setStyleSheet("")

    ub_view.a_line.setText("18.26")
    ub_view.b_line.setText("18.59")
    ub_view.c_line.setText("6.56")
    ub_view.alpha_line.setText("90")
    ub_view.beta_line.setText("90")
    ub_view.gamma_line.setText("90")

    ub_view.a_line.setStyleSheet("background-color: yellow;")
    ub_view.b_line.setStyleSheet("background-color: yellow;")
    ub_view.c_line.setStyleSheet("background-color: yellow;")
    ub_view.alpha_line.setStyleSheet("background-color: yellow;")
    ub_view.beta_line.setStyleSheet("background-color: yellow;")
    ub_view.gamma_line.setStyleSheet("background-color: yellow;")
    ub_view.set_ub_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.set_ub_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Natrolite_UB_set.png"), "png"
    )

    ub_view.a_line.setStyleSheet("")
    ub_view.b_line.setStyleSheet("")
    ub_view.c_line.setStyleSheet("")
    ub_view.alpha_line.setStyleSheet("")
    ub_view.beta_line.setStyleSheet("")
    ub_view.gamma_line.setStyleSheet("")
    ub_view.optimize_combo.setStyleSheet("")
    ub_view.conventional_button.setStyleSheet("")
    ub_view.refine_button.setStyleSheet("")

    ub_view.tab_widget.setCurrentIndex(1)
    ub_view.peaks_table.selectRow(0)

    ub_view.h1_line.setText("2")
    ub_view.k1_line.setText("2")
    ub_view.l1_line.setText("0")

    ub_view.int_h_line.setText("2")
    ub_view.int_k_line.setText("2")
    ub_view.int_l_line.setText("0")
    ub_presenter.hand_index_integer()

    ub_view.h1_line.setStyleSheet("background-color: yellow;")
    ub_view.k1_line.setStyleSheet("background-color: yellow;")
    ub_view.l1_line.setStyleSheet("background-color: yellow;")
    ub_view.highlight_1_button.setStyleSheet("background-color: green;")
    ub_view.calculate_button.setStyleSheet("background-color: green;")

    ub_view.int_h_line.setStyleSheet("background-color: yellow;")
    ub_view.int_k_line.setStyleSheet("background-color: yellow;")
    ub_view.int_l_line.setStyleSheet("background-color: yellow;")

    QTest.mouseClick(ub_view.highlight_1_button, Qt.LeftButton)
    QTest.mouseClick(ub_view.calculate_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Natrolite_UB_index_first_peak.png"), "png"
    )

    ub_view.h1_line.setStyleSheet("")
    ub_view.k1_line.setStyleSheet("")
    ub_view.l1_line.setStyleSheet("")
    ub_view.highlight_1_button.setStyleSheet("")
    ub_view.calculate_button.setStyleSheet("")

    ub_view.peaks_table.clearSelection()
    ub_view.peaks_table.selectRow(1)

    ub_view.h2_line.setText("1")
    ub_view.k2_line.setText("-5")
    ub_view.l2_line.setText("1")

    ub_view.int_h_line.setText("1")
    ub_view.int_k_line.setText("-5")
    ub_view.int_l_line.setText("1")
    ub_presenter.hand_index_integer()

    ub_view.h2_line.setStyleSheet("background-color: yellow;")
    ub_view.k2_line.setStyleSheet("background-color: yellow;")
    ub_view.l2_line.setStyleSheet("background-color: yellow;")
    ub_view.highlight_2_button.setStyleSheet("background-color: green;")
    ub_view.calculate_button.setStyleSheet("background-color: green;")
    ub_view.calculate_highlight_button.setStyleSheet(
        "background-color: green;"
    )

    QTest.mouseClick(ub_view.highlight_2_button, Qt.LeftButton)
    QTest.mouseClick(ub_view.calculate_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Natrolite_UB_index_second_peak.png"), "png"
    )

    ub_view.h2_line.setStyleSheet("")
    ub_view.k2_line.setStyleSheet("")
    ub_view.l2_line.setStyleSheet("")
    ub_view.highlight_2_button.setStyleSheet("")
    ub_view.calculate_button.setStyleSheet("")
    ub_view.calculate_highlight_button.setStyleSheet("")

    ub_view.int_h_line.setStyleSheet("")
    ub_view.int_k_line.setStyleSheet("")
    ub_view.int_l_line.setStyleSheet("")

    ub_view.peaks_table.clearSelection()

    ub_view.tab_widget.setCurrentIndex(0)
    ub_view.ub_tab.setCurrentIndex(2)

    index = ub_view.optimize_combo.findText("Constrained")
    ub_view.optimize_combo.setCurrentIndex(index)

    ub_view.optimize_combo.setStyleSheet("background-color: yellow;")
    ub_view.refine_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.refine_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Natrolite_UB_calculate_UB.png"), "png"
    )

    ub_view.optimize_combo.setStyleSheet("")
    ub_view.refine_button.setStyleSheet("")

    ub_view.peaks_tab.setCurrentIndex(1)
    ub_view.index_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.index_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Natrolite_UB_index_peaks.png"), "png"
    )

    ub_view.index_button.setStyleSheet("")

    ub_view.peaks_tab.setCurrentIndex(2)
    ub_view.predict_button.setStyleSheet("background-color: green;")

    index = ub_view.centering_combo.findText("F")
    ub_view.centering_combo.setCurrentIndex(index)
    ub_view.centering_combo.setStyleSheet("background-color: yellow;")

    QTest.mouseClick(ub_view.predict_button, Qt.LeftButton)
    QTest.qWait(1000 * 15)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Natrolite_UB_predict_peaks.png"), "png"
    )

    ub_view.predict_button.setStyleSheet("")
    ub_view.centering_combo.setStyleSheet("")

    ub_view.peaks_tab.setCurrentIndex(3)
    ub_view.integrate_button.setStyleSheet("background-color: green;")

    ub_view.adaptive_box.setChecked(True)
    ub_view.centroid_box.setChecked(True)
    ub_view.adaptive_box.setStyleSheet("background-color: yellow;")
    ub_view.centroid_box.setStyleSheet("background-color: yellow;")

    QTest.mouseClick(ub_view.integrate_button, Qt.LeftButton)
    QTest.qWait(1000 * 15)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Natrolite_UB_integrate_peaks.png"), "png"
    )

    ub_view.integrate_button.setStyleSheet("")
    ub_view.adaptive_box.setStyleSheet("")
    ub_view.centroid_box.setStyleSheet("")

    ub_view.peaks_tab.setCurrentIndex(4)
    ub_view.filter_button.setStyleSheet("background-color: green;")

    ub_view.filter_line.setText("20")
    ub_view.filter_line.setStyleSheet("background-color: yellow;")

    QTest.mouseClick(ub_view.filter_button, Qt.LeftButton)
    QTest.qWait(1000 * 15)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Natrolite_UB_filter_peaks.png"), "png"
    )

    ub_view.filter_button.setStyleSheet("")
    ub_view.filter_line.setStyleSheet("")

    ub_view.ub_tab.setCurrentIndex(2)

    index = ub_view.optimize_combo.findText("Orthorhombic")
    ub_view.optimize_combo.setCurrentIndex(index)

    ub_view.optimize_combo.setStyleSheet("background-color: yellow;")
    ub_view.refine_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ub_view.refine_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_UB_refine_UB.png"), "png"
    )

    ub_view.optimize_combo.setStyleSheet("")
    ub_view.refine_button.setStyleSheet("")

    ub_view.tab_widget.setCurrentIndex(1)
    ub_view.peaks_table.selectRow(0)

    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Natrolite_UB_view_peak.png"), "png"
    )

    ub_view.peaks_table.clearSelection()

    ub_view.tab_widget.setCurrentIndex(2)
    ub_view.inspect_verify_tab.setCurrentIndex(0)

    ub_view.slice_combo.setStyleSheet("background-color: yellow;")
    ub_view.slice_line.setStyleSheet("background-color: yellow;")
    ub_view.convert_to_hkl_button.setStyleSheet("background-color: yellow;")

    QTest.mouseClick(ub_view.convert_to_hkl_button, Qt.LeftButton)
    QTest.qWait(1000 * 15)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Natrolite_UB_slice_view.png"), "png"
    )

    ub_view.slice_combo.setStyleSheet("")
    ub_view.slice_line.setStyleSheet("")
    ub_view.convert_to_hkl_button.setStyleSheet("")

    ub_view.tab_widget.setCurrentIndex(0)

    ub_view.save_ub_button.setStyleSheet("background-color: yellow;")

    QTest.qWait(1000 * 1)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Natrolite_UB_save_UB.png"), "png"
    )
    ub_presenter.model.save_UB(
        os.path.join("/SNS/CORELLI/IPTS-31429/shared/nxv/", "Natrolite_UB.mat")
    )

    copy_generated_pngs(directory)


SCENARIOS = {
    "TOPAZ_Si_UB": TOPAZ_Si_UB,
    "TOPAZ_Scolecite_UB": TOPAZ_Scolecite_UB,
    "CORELLI_Bixbyite_UB": CORELLI_Bixbyite_UB,
    "MANDI_Mesolite_UB": MANDI_Mesolite_UB,
    "CORELLI_Natrolite_UB": CORELLI_Natrolite_UB,
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name not in SCENARIOS:
            print(f"Unknown scenario: {name}")
            print(f"Available: {', '.join(SCENARIOS)}")
            sys.exit(1)
        run_qt_scenario(SCENARIOS[name])
    else:
        script = os.path.abspath(__file__)
        failed = []
        for name in SCENARIOS:
            print(f"Running {name} ...")
            rc = subprocess.run([sys.executable, script, name]).returncode
            if rc != 0:
                failed.append(name)
        if failed:
            print(f"Failed: {', '.join(failed)}")
            sys.exit(1)
