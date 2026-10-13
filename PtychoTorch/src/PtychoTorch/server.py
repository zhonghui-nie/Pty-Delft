import io
import itertools
import json
import socket
import threading
from collections.abc import Iterable, Mapping
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
from tifffile import tifffile
from torch._prims_common import Tensor

from .engine import AutoGradPIE
from .typing import Roi2D


def is_port_in_use(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def create_numpy_tiff(
    images: torch.Tensor, phase: bool = False
) -> tuple[np.ndarray, list[str]]:
    ret: list[Tensor] = []
    labels: list[str] = []

    shape = images.shape[:-2]
    indices = itertools.product(*(range(dim) for dim in shape))

    for idx in indices:
        img = images[idx]
        if phase:
            ret.append(img.angle())
        else:
            ret.append(img.abs())

        labels.append(f"index: {idx}")

    return torch.stack(ret).detach().cpu().numpy(), labels


def get_http_request_handler_class(
    engine: list[AutoGradPIE],
    roi_select: list[Roi2D],
    rois_named: dict[str, Roi2D],
) -> type[SimpleHTTPRequestHandler]:
    class MyHttpRequestHandler(SimpleHTTPRequestHandler):
        url_probe_abs = "/api/v1/probe/abs.tif"
        url_probe_phase = "/api/v1/probe/phase.tif"

        url_object_abs = "/api/v1/object/abs.tif"
        url_object_phase = "/api/v1/object/phase.tif"

        # json
        url_reconstruction = "/api/v1/reconstruction"
        url_rect_roi = "/api/v1/rect_roi"
        url_all_roi = "/api/v1/all_rois"
        url_loss = "/api/v1/loss"

        url_shutdown = "/api/v1/shutdown"

        def get_engine(self) -> AutoGradPIE | None:
            if len(engine) == 0:
                return None
            return engine[0]

        def get_body_dict(self):
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            body_dict = json.loads(body)
            return body_dict

        def do_POST(self):
            match urlparse(self.path).path:
                case self.url_rect_roi:
                    self.POST_rect_roi()
                case self.url_all_roi:
                    self.POST_all_rois()
                case self.url_shutdown:
                    self.send_response(200)
                    self.end_headers()
                    self.server.shutdown()
                    self.server.server_close()
                    print("server stopped")
                case _:
                    self.serve_not_found()

        def do_GET(self):
            engine = self.get_engine()
            if engine is None:
                self.serve_not_found()
                return

            match urlparse(self.path).path:
                case self.url_probe_abs:
                    self.GET_ProbeObject(engine.rec_probe)

                case self.url_probe_phase:
                    self.GET_ProbeObject(engine.rec_probe, phase=True)

                case self.url_object_abs:
                    self.GET_ProbeObject(engine.rec_object)

                case self.url_object_phase:
                    self.GET_ProbeObject(engine.rec_object, phase=True)

                case self.url_reconstruction:
                    self.GET_reconstruction()

                case self.url_loss:
                    self.GET_loss()

                case self.url_all_roi:
                    self.GET_all_rois()

                case _:
                    self.serve_not_found()

        def POST_rect_roi(self):
            self.start_headers()

            body_dict = self.get_body_dict()

            x = body_dict["x"]
            y = body_dict["y"]
            h = body_dict["h"]
            w = body_dict["w"]

            roi_select.append((slice(y, y + h), slice(x, x + w)))

        def GET_all_rois(self):
            self.start_headers({"Content-type": "application/json"})

            response_dict = {}
            for roi_name, roi in rois_named.items():
                response_dict[roi_name] = {
                    "x": roi[-1].start,
                    "y": roi[-2].start,
                    "w": roi[-1].stop - roi[-1].start,
                    "h": roi[-2].stop - roi[-2].start,
                }

            with io.BytesIO() as buf:
                buf.write(json.dumps(response_dict).encode("utf-8"))
                self.wfile.write(buf.getbuffer())

        def POST_all_rois(self):
            self.start_headers()

            rois_named.clear()
            for roi_name, roi_dict in self.get_body_dict().items():
                x = roi_dict["x"]
                y = roi_dict["y"]
                h = roi_dict["h"]
                w = roi_dict["w"]
                rois_named[roi_name] = (slice(y, y + h), slice(x, x + w))

        def GET_ProbeObject(self, images: torch.Tensor | None, phase: bool = False):
            eng = self.get_engine()

            if images is None:
                self.serve_not_found()
                return

            if eng is None:
                self.serve_not_found()
                return

            self.start_headers({"Content-type": "image/tiff"})

            with io.BytesIO() as buf:
                arr, labels = create_numpy_tiff(images=images, phase=phase)

                dx_um = eng.propagator.dx * 1e6
                resolution = (1 / dx_um, 1 / dx_um)
                tifffile.imwrite(
                    buf,
                    arr,
                    metadata={"Labels": labels},
                    imagej=True,
                    resolution=resolution,
                )
                self.wfile.write(buf.getbuffer())

        def GET_reconstruction(self):
            self.start_headers({"Content-type": "application/json"})

            eng = self.get_engine()
            if eng is None:
                self.serve_not_found()
                return
            response = {
                "is_reconstructing": eng.is_reconstructing,
                # "pixel_size": self.get_engine().propagator.dx,
            }

            with io.BytesIO() as buf:
                buf.write(json.dumps(response).encode("utf-8"))
                self.wfile.write(buf.getbuffer())

        def GET_loss(self):
            self.start_headers({"Content-type": "application/json"})
            print("get loss")

            eng = self.get_engine()
            if eng is None:
                self.serve_not_found()
                return

            response = {
                "loss": eng.loss,
            }

            with io.BytesIO() as buf:
                buf.write(json.dumps(response).encode("utf-8"))
                self.wfile.write(buf.getbuffer())

        def serve_not_found(self):
            self.send_response(404)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>404 Not Found</h1></body></html>")

        def start_headers(self, header_dict: dict[str, str] | None = None):
            if header_dict is None:
                header_dict = {}
            self.send_response(200)
            for k, v in header_dict.items():
                self.send_header(k, v)
            self.end_headers()

        def end_headers(self):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

        def log_message(self, format: str, *args: Any):
            # Override to prevent logging to console
            pass

    return MyHttpRequestHandler


class HTTP_Server(threading.Thread):
    def __init__(
        self,
        engine: AutoGradPIE | None = None,
        port: int = 8080,
        host: str = "",
        auto_start: bool = True,
        *args: Iterable[Any],
        **kwargs: Mapping[str, Any] | None,
    ):
        self._engine: list[AutoGradPIE] = [engine] if engine else []
        self._roi_select: list[Roi2D] = []
        self._rois_named: dict[str, Roi2D] = {}
        self.PORT = port
        self.HOST = host or socket.gethostbyname(socket.getfqdn())

        if is_port_in_use(self.HOST, self.PORT):
            raise Exception(f"PORT: {self.PORT} already in use")

        super().__init__(args=args, kwargs=kwargs)

        if auto_start:
            self.start()

    @property
    def engine(self) -> AutoGradPIE:
        if len(self._engine) == 0:
            raise Exception("No engine set for server")
        return self._engine[0]

    @engine.setter
    def engine(self, new_engine: AutoGradPIE):
        if len(self._engine) == 0:
            self._engine.append(new_engine)
        else:
            self._engine[0] = new_engine

    @property
    def roi_select(self) -> list[Roi2D]:
        return self._roi_select

    @roi_select.setter
    def roi_select(self, new_value: list[Roi2D]):
        while self._roi_select:
            self._roi_select.pop()

        for el in new_value:
            self._roi_select.append(el)

    @property
    def rois_named(self) -> dict[str, Roi2D]:
        return self._rois_named

    @rois_named.setter
    def rois_named(self, new_value: dict[str, Roi2D]):
        while self._rois_named:
            self._rois_named.popitem()

        for k, v in new_value.items():
            self._rois_named[k] = v

    def run(self):
        hdlr = get_http_request_handler_class(
            self._engine,
            self._roi_select,
            self._rois_named,
        )
        self.request_handler = hdlr

        self.server = ThreadingHTTPServer((self.HOST, self.PORT), hdlr)

        url = f"http://{self.HOST}:{self.PORT}"
        print("server started")
        print("don't forget to add firewall exception")
        print(f"serving at: {url}")

        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        print("server stopped")
