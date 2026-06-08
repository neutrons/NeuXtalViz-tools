import sys
import traceback
import contextlib
import io
import logging
import threading

from qtpy.QtCore import QRunnable, QThreadPool, Signal, QObject, Slot


class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(tuple)
    progress = Signal(str, int)
    result = Signal(object)
    output = Signal(str)


class EmittingStream(io.StringIO):
    def __init__(self, emit_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emit_func = emit_func

    def write(self, s):
        super().write(s)
        self.emit_func(s)

    def flush(self):
        pass


class SignalLogHandler(logging.Handler):
    def __init__(self, emit_func):
        super().__init__()
        self.emit_func = emit_func

    def emit(self, record):
        msg = self.format(record)
        self.emit_func(msg)


class Worker(QRunnable):
    """
    Worker thread for running tasks in the background.

    The worker automatically passes 'progress' and 'stop_event' to the task function.

    Example task function that can be stopped:

        def my_task(progress=None, stop_event=None, **kwargs):
            for i in range(100):
                # Check if stop was requested
                if stop_event and stop_event.is_set():
                    print("Task stopped by user")
                    return None

                # Do work...
                time.sleep(0.1)

                # Report progress
                if progress:
                    progress(f"Processing step {i+1}", (i+1))

            return result
    """

    def __init__(self, task, *args, **kwargs):
        super().__init__()

        self.signals = WorkerSignals()
        self.task = task
        self.args = args
        self.kwargs = kwargs

        self.stop_event = threading.Event()
        self.kwargs["progress"] = self.emit_progress
        self.kwargs["stop_event"] = self.stop_event

    @Slot()
    def run(self):
        if sys.excepthook is None:
            sys.excepthook = sys.__excepthook__

        def emit_to_signal(s):
            if s:
                self.signals.output.emit(s)

        out_stream = EmittingStream(emit_to_signal)
        err_stream = EmittingStream(emit_to_signal)
        log_handler = SignalLogHandler(emit_to_signal)
        log_handler.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(log_handler)
        try:
            with (
                contextlib.redirect_stdout(out_stream),
                contextlib.redirect_stderr(err_stream),
            ):
                try:
                    result = self.task(*self.args, **self.kwargs)
                except:
                    traceback.print_exc()
                    exctype, value = sys.exc_info()[:2]
                    self.signals.error.emit(
                        (exctype, value, traceback.format_exc())
                    )
                else:
                    self.signals.result.emit(result)
                finally:
                    self.signals.finished.emit()
        finally:
            root_logger.removeHandler(log_handler)

    def emit_progress(self, message, progress):
        self.signals.progress.emit(message, progress)

    def connect_result(self, process):
        self.signals.result.connect(process)

    def connect_finished(self, process):
        self.signals.finished.connect(process)

    def connect_progress(self, process):
        self.signals.progress.connect(process)

    def stop(self):
        """Request the worker to stop by setting the stop event."""
        self.stop_event.set()

    def is_stopped(self):
        """Check if stop has been requested."""
        return self.stop_event.is_set()


class ThreadPool(QThreadPool):
    """
    Thread pool for managing worker threads.

    Tracks active workers and provides functionality to stop all running processes.
    """

    def __init__(self):
        super().__init__()
        self.active_workers = []

    def start_worker_pool(self, worker):
        """Start a worker and track it in the active workers list."""
        self.active_workers.append(worker)
        worker.signals.finished.connect(lambda: self.remove_worker(worker))
        self.start(worker)

    def remove_worker(self, worker):
        """Remove worker from active list when finished."""
        if worker in self.active_workers:
            self.active_workers.remove(worker)

    def stop_all_workers(self):
        """Stop all active workers by setting their stop events."""
        for worker in self.active_workers[:]:
            worker.stop()
