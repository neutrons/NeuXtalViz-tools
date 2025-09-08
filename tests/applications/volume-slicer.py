import os

from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
from qtpy.QtWidgets import QTableWidgetItem

from utilities import run_qt_scenario, copy_generated_pngs

DIRECTORY = os.path.dirname(os.path.abspath(__file__))


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


if __name__ == "__main__":
    run_qt_scenario(TOPAZ_Si_volume)
    # run_qt_scenario(CORELLI_Bixbyite_volume)
