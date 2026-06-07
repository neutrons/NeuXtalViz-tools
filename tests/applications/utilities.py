import os
import glob
import shutil
import traceback
import faulthandler

faulthandler.enable()

os.environ.setdefault("QT_API", "pyside6")
# os.environ.setdefault("QT_OPENGL", "software")

from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QTimer
from qtpy.QtTest import QTest

import qdarkstyle
from qdarkstyle.light.palette import LightPalette

from NeuXtalViz.application import NeuXtalViz

"""
QT_API=pyside6 \
QT_OPENGL=software \
LIBGL_ALWAYS_SOFTWARE=1 \
MESA_GL_VERSION_OVERRIDE=3.3 \
xvfb-run -s "-screen 0 1920x1080x24" \
python tests/applications/ub-tools.py
"""


def run_qt_scenario(scenario):
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    app.setQuitOnLastWindowClosed(False)

    app.setStyleSheet(
        qdarkstyle.load_stylesheet(
            qt_api=os.environ["QT_API"],
            palette=LightPalette,
        )
    )

    window = NeuXtalViz()
    window.show()

    result = {"rc": 0}

    def _wrapped():
        try:
            scenario(app, window)
            result["rc"] = 0

        except BaseException:
            traceback.print_exc()
            result["rc"] = 1

        finally:
            # Let pending screenshot/file/UI events flush.
            for _ in range(10):
                app.processEvents()
                QTest.qWait(50)

            # Do NOT close/delete the window.
            # Do NOT kill QThreads here.
            # Do NOT call deleteLater().
            #
            # For Qt6 + VTK/PyVista under xvfb, the segfault is often in
            # C++ destructor order during QApplication teardown.
            app.exit(result["rc"])

    QTimer.singleShot(0, _wrapped)

    rc = app.exec()

    # Critical: bypass Python/Qt/VTK destructor cleanup.
    os._exit(rc)


def copy_generated_pngs(directory):
    static = os.path.abspath(os.path.join(directory, "../../docs/source"))
    os.makedirs(static, exist_ok=True)
    for png in glob.glob(
        os.path.join(directory, "**", "*.png"), recursive=True
    ):
        shutil.copy2(png, static)
