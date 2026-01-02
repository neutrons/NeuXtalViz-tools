import os

from qtpy.QtTest import QTest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QTableWidgetItem

from utilities import run_qt_scenario, copy_generated_pngs

DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def TOPAZ_Si_plan(app, window):
    directory = os.path.join(DIRECTORY, "TOPAZ")

    app_stack = window.centralWidget().layout().itemAt(0).widget()
    app_stack.setCurrentIndex(1)

    ep_presenter = window.ep
    ep_view = ep_presenter.view

    ep_presenter.model.load_UB(
        os.path.join("/SNS/TOPAZ/IPTS-36169/shared/nxv/", "Si_UB.mat")
    )
    ep_presenter.update_oriented_lattice()
    ep_presenter.view.set_transform(ep_presenter.model.get_transform())

    index = ep_view.instrument_combo.findText("TOPAZ")
    ep_view.instrument_combo.setCurrentIndex(index)

    index = ep_view.mode_combo.findText("Ambient")
    ep_view.mode_combo.setCurrentIndex(index)

    ep_view.instrument_combo.setStyleSheet("background-color: yellow;")
    ep_view.mode_combo.setStyleSheet("background-color: yellow;")
    ep_view.load_UB_button.setStyleSheet("background-color: yellow;")

    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_plan_initialize.png"), "png"
    )

    ep_view.instrument_combo.setStyleSheet("")
    ep_view.mode_combo.setStyleSheet("")
    ep_view.load_UB_button.setStyleSheet("")

    ep_view.goniometer_table.selectRow(0)
    ep_view.goniometer_table.setItem(0, 1, QTableWidgetItem("-90"))
    ep_view.goniometer_table.setItem(0, 2, QTableWidgetItem("90"))

    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_plan_update.png"), "png"
    )

    ep_view.goniometer_table.clearSelection()

    ep_view.tab_widget.setCurrentIndex(1)

    ep_view.h1_line.setText("4")
    ep_view.k1_line.setText("0")
    ep_view.l1_line.setText("0")

    ep_view.h1_line.setStyleSheet("background-color: yellow;")
    ep_view.k1_line.setStyleSheet("background-color: yellow;")
    ep_view.l1_line.setStyleSheet("background-color: yellow;")
    ep_view.calculate_single_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ep_view.calculate_single_button, Qt.LeftButton)
    QTest.qWait(1000 * 40)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_plan_calculate_peak.png"), "png"
    )

    ep_view.h1_line.setStyleSheet("")
    ep_view.k1_line.setStyleSheet("")
    ep_view.l1_line.setStyleSheet("")
    ep_view.calculate_single_button.setStyleSheet("")

    ep_view.add_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ep_view.canvas_inst, Qt.LeftButton)
    QTest.qWait(1000 * 10)
    QTest.mouseClick(ep_view.add_button, Qt.LeftButton)
    QTest.qWait(1000 * 10)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_plan_add_orientation.png"), "png"
    )

    ep_view.add_button.setStyleSheet("")

    ep_view.tab_widget.setCurrentIndex(0)
    ep_view.values_tab.setCurrentIndex(2)

    index = ep_view.crystal_combo.findText("Cubic")
    ep_view.crystal_combo.setCurrentIndex(index)
    ep_presenter.switch_crystal()

    index = ep_view.point_group_combo.findText("m-3m")
    ep_view.point_group_combo.setCurrentIndex(index)

    index = ep_view.lattice_centering_combo.findText("F")
    ep_view.lattice_centering_combo.setCurrentIndex(index)

    ep_view.settings_line.setText("3")

    ep_view.crystal_combo.setStyleSheet("background-color: yellow;")
    ep_view.point_group_combo.setStyleSheet("background-color: yellow;")
    ep_view.lattice_centering_combo.setStyleSheet("background-color: yellow;")
    ep_view.settings_line.setStyleSheet("background-color: yellow;")
    ep_view.optimize_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ep_view.optimize_button, Qt.LeftButton)
    QTest.qWait(1000 * 30)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_plan_optimize_coverage.png"), "png"
    )

    ep_view.crystal_combo.setStyleSheet("")
    ep_view.point_group_combo.setStyleSheet("")
    ep_view.lattice_centering_combo.setStyleSheet("")
    ep_view.settings_line.setStyleSheet("")
    ep_view.optimize_button.setStyleSheet("")

    ep_view.title_line.setText("Si plan")

    ep_view.title_line.setStyleSheet("background-color: yellow;")
    ep_view.highlight_button.setStyleSheet("background-color: yellow;")
    ep_view.update_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(ep_view.highlight_button, Qt.LeftButton)
    QTest.qWait(1000 * 10)
    QTest.mouseClick(ep_view.update_button, Qt.LeftButton)
    QTest.qWait(1000 * 10)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_plan_update_info.png"), "png"
    )

    ep_view.title_line.setStyleSheet("")
    ep_view.highlight_button.setStyleSheet("")
    ep_view.update_button.setStyleSheet("")

    ep_view.save_plan_button.setStyleSheet("background-color: yellow;")

    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_plan_save_table_scan.png"), "png"
    )

    copy_generated_pngs(directory)


if __name__ == "__main__":
    run_qt_scenario(TOPAZ_Si_plan)
    # run_qt_scenario(CORELLI_Bixbyite_plan)
