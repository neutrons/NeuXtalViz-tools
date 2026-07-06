import sys
import traceback
import contextlib
import io
import logging
import threading

from qtpy.QtCore import QRunnable, QThreadPool, Signal, QObject, Slot


class WorkerSignals(QObject):
    """
    Collection of signals used to communicate :class:`Worker` state.

    Attributes
    ----------
    finished : Signal
        Emitted with no arguments when the worker's task has finished
        running, regardless of success or failure.
    error : Signal(tuple)
        Emitted with a ``(exctype, value, traceback_str)`` tuple when the
        task raises an exception.
    progress : Signal(str, int)
        Emitted with a status message and a progress value as the task
        reports progress.
    result : Signal(object)
        Emitted with the return value of the task once it completes
        successfully.
    output : Signal(str)
        Emitted with captured stdout/stderr/log text produced by the task.
    """

    finished = Signal()
    error = Signal(tuple)
    progress = Signal(str, int)
    result = Signal(object)
    output = Signal(str)


class EmittingStream(io.StringIO):
    """
    In-memory text stream that forwards written text to a callback.

    Used to redirect stdout/stderr from a background task so that the
    written text can be relayed to the GUI via a Qt signal.
    """

    def __init__(self, emit_func, *args, **kwargs):
        """
        Initialize the stream with a callback to receive written text.

        Parameters
        ----------
        emit_func : callable
            Function called with each string written to the stream.
        *args : tuple
            Additional positional arguments passed to
            :class:`io.StringIO`.
        **kwargs : dict
            Additional keyword arguments passed to :class:`io.StringIO`.
        """
        super().__init__(*args, **kwargs)
        self.emit_func = emit_func

    def write(self, s):
        """
        Write text to the underlying buffer and forward it to the callback.

        Parameters
        ----------
        s : str
            Text to write.

        Returns
        -------
        n : int
            Number of characters written, as returned by
            :meth:`io.StringIO.write`.
        """
        super().write(s)
        self.emit_func(s)

    def flush(self):
        """Do nothing; flushing is a no-op for this stream."""
        pass


class SignalLogHandler(logging.Handler):
    """
    Logging handler that forwards formatted log records to a callback.

    Used to relay log messages emitted during a background task to the
    GUI via a Qt signal.
    """

    def __init__(self, emit_func):
        """
        Initialize the handler with a callback to receive formatted records.

        Parameters
        ----------
        emit_func : callable
            Function called with each formatted log message.
        """
        super().__init__()
        self.emit_func = emit_func

    def emit(self, record):
        """
        Format a log record and forward it to the callback.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to format and emit.
        """
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
        """
        Initialize the worker with a task and its arguments.

        Parameters
        ----------
        task : callable
            Function to run in the background. It will be called as
            ``task(*args, progress=self.emit_progress,
            stop_event=self.stop_event, **kwargs)``.
        *args : tuple
            Positional arguments to pass to `task`.
        **kwargs : dict
            Keyword arguments to pass to `task`. The keys ``"progress"``
            and ``"stop_event"`` are added/overwritten automatically.
        """
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
        """
        Run the task, redirecting stdout/stderr/logging to Qt signals.

        Executes `self.task` with the stored arguments, capturing any
        printed output and log records and relaying them via
        ``self.signals.output``. Emits ``self.signals.result`` with the
        task's return value on success, ``self.signals.error`` with
        exception information on failure, and always emits
        ``self.signals.finished`` afterward.
        """
        if sys.excepthook is None:
            sys.excepthook = sys.__excepthook__

        def emit_to_signal(s):
            """
            Relay a non-empty string to the worker's output signal.

            Parameters
            ----------
            s : str
                Text captured from stdout/stderr or a log record.
            """
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
        """
        Emit a progress update via the worker's progress signal.

        Parameters
        ----------
        message : str
            Status message describing the current progress.
        progress : int
            Progress value (e.g. percent complete or step count).
        """
        self.signals.progress.emit(message, progress)

    def connect_result(self, process):
        """
        Connect a callback to the worker's result signal.

        Parameters
        ----------
        process : callable
            Function called with the task's return value when the
            result signal is emitted.
        """
        self.signals.result.connect(process)

    def connect_finished(self, process):
        """
        Connect a callback to the worker's finished signal.

        Parameters
        ----------
        process : callable
            Function called with no arguments when the finished signal
            is emitted.
        """
        self.signals.finished.connect(process)

    def connect_progress(self, process):
        """
        Connect a callback to the worker's progress signal.

        Parameters
        ----------
        process : callable
            Function called with the status message and progress value
            when the progress signal is emitted.
        """
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
        """Initialize the thread pool with an empty active-worker list."""
        super().__init__()
        self.active_workers = []

    def start_worker_pool(self, worker):
        """
        Start a worker and track it in the active workers list.

        Parameters
        ----------
        worker : Worker
            Worker to run in the thread pool. It is appended to
            `active_workers` and automatically removed once its
            finished signal is emitted.
        """
        self.active_workers.append(worker)
        worker.signals.finished.connect(lambda: self.remove_worker(worker))
        self.start(worker)

    def remove_worker(self, worker):
        """
        Remove worker from active list when finished.

        Parameters
        ----------
        worker : Worker
            Worker to remove from `active_workers`, if present.
        """
        if worker in self.active_workers:
            self.active_workers.remove(worker)

    def stop_all_workers(self):
        """Stop all active workers by setting their stop events."""
        for worker in self.active_workers[:]:
            worker.stop()
