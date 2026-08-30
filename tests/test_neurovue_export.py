"""The NeuroVue hand-off. NeuroVue itself is not modified, so this tests our side.

The upload path is checked against a stub server implementing the contract
NeuroVue's `POST /api/lesions` declares (multipart `file` + `name` + `base`).
That verifies the client without starting someone else's FastAPI app or its
MongoDB -- and the stub also asserts that what goes over the wire carries no
patient identifier, which is the part that actually matters here.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from lesionconn import config, deid, neurovue

pytestmark = pytest.mark.skipif(
    not config.MANIFEST_PATH.exists(),
    reason="no cohort manifest -- run scripts/13_build_cohort.py",
)


@pytest.fixture(scope="module")
def subjects():
    from lesionconn import cohort

    return cohort.load_manifest()["subject_id"].head(2).tolist()


# --- the file drop ----------------------------------------------------------


def test_bundle_matches_the_layout_neurovue_writes_itself(subjects, tmp_path):
    """LESION_DIR/<slug>/lesion_<timestamp>.nii.gz -- see backend/server.py."""
    dest = neurovue.export_bundle(subjects, tmp_path)
    for subject_id in subjects:
        subject_dir = dest / subject_id
        assert subject_dir.is_dir()
        masks = list(subject_dir.glob("lesion_*.nii.gz"))
        assert len(masks) == 1
        assert masks[0].stat().st_size > 0
        assert re.fullmatch(r"lesion_\d{8}T\d{6}Z\.nii\.gz", masks[0].name)


def test_bundle_carries_the_disconnection_profile_and_the_disclaimer(subjects, tmp_path):
    dest = neurovue.export_bundle(subjects, tmp_path)
    index = json.loads((dest / "index.json").read_text(encoding="utf-8"))
    assert index["n_subjects"] == len(subjects)
    assert index["space"]["orientation"] == "LAS"
    assert "not a diagnostic device" in index["disclaimer"].lower()

    profile = json.loads(
        (dest / subjects[0] / "disconnection.json").read_text(encoding="utf-8")
    )
    assert profile["subject_id"] == subjects[0]
    assert "not a diagnostic device" in profile["disclaimer"].lower()


def test_nothing_in_the_bundle_names_a_patient(subjects, tmp_path):
    if not config.SUBJECT_MAP_PATH.exists():
        pytest.skip("no subject map")
    dest = neurovue.export_bundle(subjects, tmp_path)
    pattern = deid.token_pattern(deid.sensitive_tokens())
    for path in dest.rglob("*"):
        if not path.is_file():
            continue
        assert not pattern.search(path.name)
        if path.suffix == ".json":
            assert not pattern.search(path.read_text(encoding="utf-8"))


def test_export_refuses_an_id_that_is_not_de_identified(tmp_path):
    with pytest.raises(ValueError, match="de-identified"):
        neurovue.export_bundle(["Jane-Doe_123456X"], tmp_path)


# --- the upload path, against a stub of NeuroVue's declared contract ---------


class _StubHandler(BaseHTTPRequestHandler):
    received: ClassVar[list[bytes]] = []

    def do_POST(self):  # BaseHTTPRequestHandler's own spelling
        if self.path != neurovue.API_PATH:
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        _StubHandler.received.append(self.rfile.read(length))
        body = json.dumps({"ok": True, "id": "stub", "bytes": length}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_server():
    _StubHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", _StubHandler
    server.shutdown()
    server.server_close()


def test_upload_sends_the_fields_neurovue_declares(subjects, stub_server):
    from lesionconn import cohort

    url, handler = stub_server
    subject_id = subjects[0]
    path = cohort.subject_paths(subject_id)["lesion_path"]

    result = neurovue.post_lesion(url, subject_id, path)
    assert result["ok"] is True
    assert len(handler.received) == 1

    body = handler.received[0]
    assert b'name="file"' in body
    assert b'name="name"' in body
    assert b'name="base"' in body
    assert subject_id.encode() in body
    assert b"\x1f\x8b" in body, "the gzipped NIfTI itself should be in the body"


def test_upload_body_carries_no_patient_identifier(subjects, stub_server):
    from lesionconn import cohort

    if not config.SUBJECT_MAP_PATH.exists():
        pytest.skip("no subject map")
    url, handler = stub_server
    subject_id = subjects[0]
    neurovue.post_lesion(url, subject_id, cohort.subject_paths(subject_id)["lesion_path"])

    body = handler.received[0]
    # the payload is a gzipped NIfTI; strip it and check the multipart text
    text_parts = b"".join(
        part for part in body.split(b"\r\n\r\n") if b"\x1f\x8b" not in part[:4]
    )
    pattern = re.compile(
        deid.token_pattern(deid.sensitive_tokens()).pattern.encode(), re.IGNORECASE
    )
    assert not pattern.search(text_parts)


def test_upload_refuses_an_id_that_is_not_de_identified(subjects, stub_server):
    from lesionconn import cohort

    url, _ = stub_server
    path = cohort.subject_paths(subjects[0])["lesion_path"]
    with pytest.raises(ValueError, match="de-identified"):
        neurovue.post_lesion(url, "Jane-Doe_123456X", path)
