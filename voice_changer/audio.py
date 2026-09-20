"""sounddevice I/O wrapper.

Two independent streams (input + output) communicate through a small queue.
The output callback pulls a block, runs it through the pipeline, and writes
to the output device. If the queue is empty (input late), we output silence
rather than block, because Meet/Zoom prefers a glitch over a stalled stream.
"""

import queue

import numpy as np
import sounddevice as sd


def list_input_devices():
    return [(i, d) for i, d in enumerate(sd.query_devices())
            if d['max_input_channels'] > 0]


def list_output_devices():
    return [(i, d) for i, d in enumerate(sd.query_devices())
            if d['max_output_channels'] > 0]


def find_default_virtual_cable(devices) -> int | None:
    """Best-effort guess for VB-CABLE Input among output devices."""
    for i, d in devices:
        name = d['name'].lower()
        if 'cable' in name and 'input' in name:
            return i
    return None


class AudioEngine:
    def __init__(self, pipeline, sample_rate=48000, block_size=1024):
        self.pipeline = pipeline
        self.sr = sample_rate
        self.block = block_size
        self._in_stream = None
        self._out_stream = None
        self._queue: queue.Queue = queue.Queue(maxsize=64)
        self._running = False
        self._last_status = ""

    @property
    def running(self) -> bool:
        return self._running

    @property
    def status(self) -> str:
        return self._last_status

    def start(self, in_dev: int, out_dev: int):
        if self._running:
            self.stop()

        def in_cb(indata, frames, t, status):
            if status:
                self._last_status = str(status)
            try:
                self._queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                # drop oldest, push newest, which keeps latency from growing
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(indata[:, 0].copy())
                except queue.Empty:
                    pass

        def out_cb(outdata, frames, t, status):
            if status:
                self._last_status = str(status)
            try:
                block = self._queue.get_nowait()
            except queue.Empty:
                outdata.fill(0)
                return
            out = self.pipeline.process(block)
            n = min(len(out), frames)
            outdata[:n, 0] = out[:n]
            if n < frames:
                outdata[n:, 0] = 0

        self._in_stream = sd.InputStream(
            samplerate=self.sr, blocksize=self.block, device=in_dev,
            channels=1, dtype='float32', callback=in_cb,
        )
        self._out_stream = sd.OutputStream(
            samplerate=self.sr, blocksize=self.block, device=out_dev,
            channels=1, dtype='float32', callback=out_cb,
        )
        self._in_stream.start()
        self._out_stream.start()
        self._running = True

    def stop(self):
        for s in (self._in_stream, self._out_stream):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception:
                    pass
        self._in_stream = None
        self._out_stream = None
        self._running = False
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
