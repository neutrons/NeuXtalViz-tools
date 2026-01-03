# -------------------------------------------------------------------------- #
# xvfb-run -s "-screen 0 1920x1080x24" python tests/applications/ub-tools.py #
# -------------------------------------------------------------------------- #
import sys
import os
import glob
import shutil

os.environ["QT_API"] = "pyqt5"

from qtpy.QtWidgets import QApplication
from qtpy.QtTest import QTest
from qtpy.QtCore import QTimer, QThread, QThreadPool

from NeuXtalViz.application import NeuXtalViz

import qdarkstyle

from qdarkstyle.light.palette import LightPalette

DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def run_qt_scenario(scenario):
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(palette=LightPalette))
    app.setQuitOnLastWindowClosed(True)

    window = NeuXtalViz()
    window.show()

    def _wrapped():
        scenario(app, window)

        window.close()
        QApplication.closeAllWindows()

        for _ in range(3):
            app.processEvents()
            QTest.qWait(50)

        for t in app.findChildren(QTimer):
            t.stop()

        for th in app.findChildren(QThread):
            th.requestInterruption()
            th.quit()
        for th in app.findChildren(QThread):
            th.wait(3000)

        QThreadPool.globalInstance().waitForDone(5000)

    QTimer.singleShot(0, app.quit)

    QTimer.singleShot(0, _wrapped)
    rc = app.exec_()

    # Previously this function forcibly terminated the process with os._exit(0)
    # if any non-main threads were still alive. That prevented multiple
    # scenarios from running sequentially (e.g. in ub-tools.py). We now
    # rely on the explicit QThread and QThreadPool cleanup above and simply
    # return the Qt exit code so callers can invoke this multiple times.
    return rc


def copy_generated_pngs(directory):
    static = os.path.abspath(os.path.join(DIRECTORY, "../../docs/source"))
    os.makedirs(static, exist_ok=True)
    for png in glob.glob(
        os.path.join(directory, "**", "*.png"), recursive=True
    ):
        shutil.copy2(png, static)
