import os
import sys
import subprocess

from qtpy.QtTest import QTest
from qtpy.QtCore import Qt

from utilities import run_qt_scenario, copy_generated_pngs

DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def _select_combo_item_starting_with(combo, prefix):
    for i in range(combo.count()):
        if combo.itemText(i).startswith(prefix):
            combo.setCurrentIndex(i)
            return
    raise ValueError(f"No combo item starting with {prefix!r}")


def TOPAZ_Si_volume(app, window):
    directory = os.path.join(DIRECTORY, "TOPAZ")

    app_stack = window.centralWidget().layout().itemAt(0).widget()
    app_stack.setCurrentIndex(2)

    vs_presenter = window.vs
    vs_view = vs_presenter.view

    vs_presenter.model.load_md_histo_workspace(
        "/SNS/TOPAZ/IPTS-36169/shared/nxv/Si_AG_300K_normalization/"
        + "Si_AG_300K_(h,k,0)_[0,0,l]_"
        + "[-10.0,10.0]_[-10.0,10.0]_[-10.0,10.0]_201x201x201_m-3m_sub_bkg.nxs"
    )
    vs_presenter.refresh_workspace_lists()
    vs_presenter.update_oriented_lattice()
    vs_presenter.view.set_transform(vs_presenter.model.get_transform())
    vs_presenter.redraw_data()

    vs_view.load_NXS_button.setStyleSheet("background-color: green;")

    QTest.qWait(1000 * 10)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_volume_load.png"), "png"
    )

    vs_view.load_NXS_button.setStyleSheet("")

    vs_view.slice_line.setText("2.0")

    vs_view.slice_combo.setStyleSheet("background-color: yellow;")
    vs_view.slice_line.setStyleSheet("background-color: yellow;")

    vs_presenter.redraw_data()
    QTest.qWait(1000 * 10)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_volume_slice.png"), "png"
    )

    vs_view.slice_combo.setStyleSheet("")
    vs_view.slice_line.setStyleSheet("")

    vs_view.cut_combo.setCurrentIndex(1)
    vs_view.cut_line.setText("4.0")

    vs_view.cut_combo.setStyleSheet("background-color: yellow;")
    vs_view.cut_line.setStyleSheet("background-color: yellow;")
    vs_view.toggle_line_box.setStyleSheet("background-color: green;")

    QTest.mouseClick(vs_view.toggle_line_box, Qt.LeftButton)
    vs_presenter.redraw_data()
    QTest.qWait(1000 * 10)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_volume_cut.png"), "png"
    )

    vs_view.cut_combo.setStyleSheet("")
    vs_view.cut_line.setStyleSheet("")
    vs_view.toggle_line_box.setStyleSheet("")

    vs_view.save_slice_button.setStyleSheet("background-color: yellow;")
    vs_view.save_cut_button.setStyleSheet("background-color: yellow;")

    QTest.qWait(1000 * 10)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_volume_save.png"), "png"
    )

    copy_generated_pngs(directory)


def CORELLI_Bixbyite_deltaPDF(app, window):
    directory = os.path.join(DIRECTORY, "CORELLI")

    app_stack = window.centralWidget().layout().itemAt(0).widget()
    app_stack.setCurrentIndex(2)

    vs_presenter = window.vs
    vs_view = vs_presenter.view

    SLICE_TAB, TRANSFORM_TAB = 0, 1

    # --- Step 1: load the 6 K data ----------------------------------
    vs_presenter.model.load_md_histo_workspace(
        "/SNS/EXAMPLES/CORELLI/IPTS-12345/shared/"
        + "bixbyite_006K_(h,k,0)_[0,0,l]_"
        + "[-10.0,10.0]_[-10.0,10.0]_[-10.0,10.0]_201x201x201_m-3.nxs",
        display_name="006K",
    )
    vs_presenter.refresh_workspace_lists()
    vs_presenter.update_oriented_lattice()
    vs_view.set_transform(vs_presenter.model.get_transform())
    vs_presenter.redraw_data()

    vs_view.load_NXS_button.setStyleSheet("background-color: green;")

    QTest.qWait(1000 * 10)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_deltaPDF_load_006K.png"), "png"
    )

    vs_view.load_NXS_button.setStyleSheet("")

    # --- Step 2: load the 300 K data ---------------------------------
    vs_presenter.model.load_md_histo_workspace(
        "/SNS/EXAMPLES/CORELLI/IPTS-12345/shared/"
        + "bixbyite_300K_(h,k,0)_[0,0,l]_"
        + "[-10.0,10.0]_[-10.0,10.0]_[-10.0,10.0]_201x201x201_m-3.nxs",
        display_name="300K",
    )
    vs_presenter.refresh_workspace_lists()
    vs_presenter.redraw_data()

    vs_view.load_NXS_button.setStyleSheet("background-color: green;")

    QTest.qWait(1000 * 10)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_deltaPDF_load_300K.png"), "png"
    )

    vs_view.load_NXS_button.setStyleSheet("")

    # --- Step 3: subtract the scaled 300 K background ----------------
    vs_view.tab_widget.setCurrentIndex(TRANSFORM_TAB)

    vs_view.combine_ws_a_combo.setCurrentText("006K")
    vs_view.combine_ws_b_combo.setCurrentText("300K")
    vs_view.combine_coeff_b_line.setText("0.95")
    vs_view.combine_output_line.setText("subtracted")

    vs_view.combine_ws_a_combo.setStyleSheet("background-color: yellow;")
    vs_view.combine_ws_b_combo.setStyleSheet("background-color: yellow;")
    vs_view.combine_coeff_b_line.setStyleSheet("background-color: yellow;")
    vs_view.combine_output_line.setStyleSheet("background-color: yellow;")
    vs_view.combine_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(vs_view.combine_button, Qt.LeftButton)
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_deltaPDF_subtract.png"), "png"
    )

    vs_view.combine_ws_a_combo.setStyleSheet("")
    vs_view.combine_ws_b_combo.setStyleSheet("")
    vs_view.combine_coeff_b_line.setStyleSheet("")
    vs_view.combine_output_line.setStyleSheet("")
    vs_view.combine_button.setStyleSheet("")

    # --- Step 4: view the subtracted data -----------------------------
    vs_view.tab_widget.setCurrentIndex(SLICE_TAB)
    vs_view.workspace_combo.setCurrentText("subtracted")
    QTest.qWait(1000 * 10)

    vs_view.vmin_line.setText("0")
    vs_view.vmax_line.setText("0.00001")
    vs_presenter.update_cvals()

    vs_view.workspace_combo.setStyleSheet("background-color: yellow;")
    vs_view.vmin_line.setStyleSheet("background-color: yellow;")
    vs_view.vmax_line.setStyleSheet("background-color: yellow;")

    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_deltaPDF_view_subtracted.png"), "png"
    )

    vs_view.workspace_combo.setStyleSheet("")
    vs_view.vmin_line.setStyleSheet("")
    vs_view.vmax_line.setStyleSheet("")

    # --- Step 5: punch the Bragg peaks --------------------------------
    vs_view.tab_widget.setCurrentIndex(TRANSFORM_TAB)

    index = vs_view.pdf_crystal_system_combo.findText("Cubic")
    vs_view.pdf_crystal_system_combo.setCurrentIndex(index)
    vs_presenter.update_pdf_space_groups()
    _select_combo_item_starting_with(vs_view.pdf_space_group_combo, "206:")

    vs_view.punch_input_combo.setCurrentText("subtracted")
    vs_view.punch_q_size_line.setText("0.25")

    vs_view.pdf_crystal_system_combo.setStyleSheet("background-color: yellow;")
    vs_view.pdf_space_group_combo.setStyleSheet("background-color: yellow;")
    vs_view.punch_input_combo.setStyleSheet("background-color: yellow;")
    vs_view.punch_q_size_line.setStyleSheet("background-color: yellow;")
    vs_view.run_punch_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(vs_view.run_punch_button, Qt.LeftButton)
    QTest.qWait(1000 * 20)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_deltaPDF_punch.png"), "png"
    )

    vs_view.pdf_crystal_system_combo.setStyleSheet("")
    vs_view.pdf_space_group_combo.setStyleSheet("")
    vs_view.punch_input_combo.setStyleSheet("")
    vs_view.punch_q_size_line.setStyleSheet("")
    vs_view.run_punch_button.setStyleSheet("")

    # --- Step 5b: view the punched data --------------------------------
    vs_view.tab_widget.setCurrentIndex(SLICE_TAB)
    vs_view.workspace_combo.setCurrentText("punched")
    QTest.qWait(1000 * 10)

    vs_presenter.update_cvals()

    vs_view.workspace_combo.setStyleSheet("background-color: yellow;")

    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_deltaPDF_view_punched.png"), "png"
    )

    vs_view.workspace_combo.setStyleSheet("")

    # --- Step 6: fill the punched regions (blur) ----------------------
    vs_view.tab_widget.setCurrentIndex(TRANSFORM_TAB)

    vs_view.blur_input_combo.setCurrentText("punched")
    vs_view.blur_output_line.setText("filled")
    vs_view.blur_q_blur_line.setText("0.1")

    vs_view.blur_input_combo.setStyleSheet("background-color: yellow;")
    vs_view.blur_output_line.setStyleSheet("background-color: yellow;")
    vs_view.blur_q_blur_line.setStyleSheet("background-color: yellow;")
    vs_view.run_blur_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(vs_view.run_blur_button, Qt.LeftButton)
    QTest.qWait(1000 * 15)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_deltaPDF_fill.png"), "png"
    )

    vs_view.blur_input_combo.setStyleSheet("")
    vs_view.blur_output_line.setStyleSheet("")
    vs_view.blur_q_blur_line.setStyleSheet("")
    vs_view.run_blur_button.setStyleSheet("")

    # --- Step 6b: view the filled data ----------------------------------
    vs_view.tab_widget.setCurrentIndex(SLICE_TAB)
    vs_view.workspace_combo.setCurrentText("filled")
    QTest.qWait(1000 * 10)

    vs_presenter.update_cvals()

    vs_view.workspace_combo.setStyleSheet("background-color: yellow;")

    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_deltaPDF_view_filled.png"), "png"
    )

    vs_view.workspace_combo.setStyleSheet("")

    # --- Step 7: calculate the 3D-delta PDF ---------------------------
    vs_view.tab_widget.setCurrentIndex(TRANSFORM_TAB)

    vs_view.pdf_input_combo.setCurrentText("filled")
    index = vs_view.pdf_window_combo.findText("Lorch")
    vs_view.pdf_window_combo.setCurrentIndex(index)

    vs_view.pdf_input_combo.setStyleSheet("background-color: yellow;")
    vs_view.pdf_window_combo.setStyleSheet("background-color: yellow;")
    vs_view.calculate_pdf_button.setStyleSheet("background-color: green;")

    QTest.mouseClick(vs_view.calculate_pdf_button, Qt.LeftButton)
    QTest.qWait(1000 * 15)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_deltaPDF_transform.png"), "png"
    )

    vs_view.pdf_input_combo.setStyleSheet("")
    vs_view.pdf_window_combo.setStyleSheet("")
    vs_view.calculate_pdf_button.setStyleSheet("")

    # --- Step 8: view the 3D-delta PDF --------------------------------
    vs_view.tab_widget.setCurrentIndex(SLICE_TAB)
    vs_view.workspace_combo.setCurrentText("pdf")
    QTest.qWait(1000 * 10)

    vs_view.vmin_line.setText("-1")
    vs_view.vmax_line.setText("1")
    vs_presenter.update_cvals()

    vs_view.workspace_combo.setStyleSheet("background-color: yellow;")
    vs_view.vmin_line.setStyleSheet("background-color: yellow;")
    vs_view.vmax_line.setStyleSheet("background-color: yellow;")

    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Bixbyite_deltaPDF_view_pdf.png"), "png"
    )

    vs_view.workspace_combo.setStyleSheet("")
    vs_view.vmin_line.setStyleSheet("")
    vs_view.vmax_line.setStyleSheet("")

    copy_generated_pngs(directory)


SCENARIOS = {
    "TOPAZ_Si_volume": TOPAZ_Si_volume,
    "CORELLI_Bixbyite_deltaPDF": CORELLI_Bixbyite_deltaPDF,
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
