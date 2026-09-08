"""
A report that outlives its request, and reaches someone without an
account.

Until now a report existed only for the duration of the HTTP response
that built it, and that single fact was behind three separate
limitations: a 3-second build had to be held on the request thread
because there was nowhere to put the result; a client could not be sent
a link, only a file, so delivering a report meant a human attaching it
to an email; and nothing could run on a schedule, because a schedule
needs somewhere to leave what it produced.

The security posture of the share link is the part worth being careful
about — it is a genuine widening of access, so these tests spend most of
their effort on what it must NOT do.
"""
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.load_control import limiter

CSV = b"region,amount\nNorth,10\nSouth,20\nEast,30\nWest,40\n" * 40


@pytest.fixture(autouse=True)
def _fresh_limiter():
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A client whose jobs, artifacts and schedules start empty.

    The three stores are module singletons that read config.data_dir
    when they are constructed at import, so setting DATA_DIR here is
    too late — without repointing them, a schedule written by one run
    is still on disk for the next one, and "exactly one report per
    window" fails against a leftover from an hour ago.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    from app.services import artifacts, jobs, report_jobs, schedules  # noqa: F401
    for module, name in ((artifacts, "artifacts"), (jobs, "jobs"),
                         (schedules, "schedules")):
        monkeypatch.setattr(module.store, "base_dir",
                            str(tmp_path / name), raising=True)

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def dataset(client):
    res = client.post("/api/datasets/upload",
                      files={"file": ("s.csv", CSV, "text/csv")})
    assert res.status_code == 200
    return res.json()["meta"]["dataset_id"]


def _finish(client, job_id, timeout_s=120):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] in ("done", "failed"):
            return state
        time.sleep(0.5)
    pytest.fail("job never finished")


# ══════════════════════════════════════════════════════════
#  Work that outlives the request
# ══════════════════════════════════════════════════════════

def test_a_job_returns_before_the_work_is_done(client, dataset):
    res = client.post("/api/jobs", json={"dataset_id": dataset,
                                         "kind": "health-report"})
    assert res.status_code == 200
    body = res.json()
    assert body["job_id"]
    assert body["status"] in ("queued", "running")


def test_the_job_finishes_and_leaves_something_behind(client, dataset):
    job = client.post("/api/jobs", json={"dataset_id": dataset,
                                         "kind": "health-report"}).json()
    done = _finish(client, job["job_id"])
    assert done["status"] == "done", done["error"]
    assert done["artifact_id"]
    assert done["duration_s"] > 0

    art = client.get(f"/api/artifacts/{done['artifact_id']}").json()
    assert art["format"] == "pdf"
    assert art["size_bytes"] > 1000

    blob = client.get(f"/api/artifacts/{done['artifact_id']}/download")
    assert blob.status_code == 200
    assert blob.content[:4] == b"%PDF"


def test_a_failure_is_a_recorded_outcome_not_a_hang(client, dataset):
    """A job that raised must end as "failed" with the reason on it. A
    background thread that dies quietly leaves a client polling forever.
    """
    from app.services.jobs import register

    def explode(job):
        raise RuntimeError("the engine fell over")

    register("boom", explode)
    job = client.post("/api/jobs", json={"dataset_id": dataset,
                                         "kind": "boom"}).json()
    done = _finish(client, job["job_id"])
    assert done["status"] == "failed"
    assert "fell over" in done["error"]
    assert not done["artifact_id"]


def test_the_full_analysis_report_runs_as_a_job_too(client, dataset):
    """Not just the health report. The analysis PDF is the long one —
    it is the reason this layer exists — and it comes back from the
    endpoint as a stream, which the runner has to be able to store."""
    job = client.post("/api/jobs", json={"dataset_id": dataset,
                                         "kind": "report"}).json()
    done = _finish(client, job["job_id"], timeout_s=180)
    assert done["status"] == "done", done["error"]

    res = client.get(f"/api/artifacts/{done['artifact_id']}/download")
    assert res.content[:4] == b"%PDF"
    assert len(res.content) > 20_000, len(res.content)


def test_a_deck_is_stored_as_a_deck_and_not_named_pdf(client, dataset):
    """The analysis job and the deck come off the same request object.
    A deck stored as .pdf reaches whoever opened the link as a file
    their machine refuses to open."""
    job = client.post("/api/jobs", json={
        "dataset_id": dataset, "kind": "report",
        "params": {"format": "pptx"}}).json()
    done = _finish(client, job["job_id"])
    assert done["status"] == "done", done["error"]

    art = client.get(f"/api/artifacts/{done['artifact_id']}").json()
    assert art["format"] == "pptx", art
    assert art["filename"].endswith(".pptx"), art

    res = client.get(f"/api/artifacts/{done['artifact_id']}/download")
    assert "presentationml" in res.headers["content-type"], \
        res.headers["content-type"]


def test_an_unknown_kind_is_refused_at_submission(client, dataset):
    res = client.post("/api/jobs", json={"dataset_id": dataset,
                                         "kind": "does-not-exist"})
    assert res.status_code == 422


def test_a_job_abandoned_by_a_restart_stops_saying_it_is_running(
        client, dataset):
    """A container restart kills the pool but not the job records. Both
    a job that was mid-run and one that was still queued belong to a
    process that is gone — and the queued one never got a start time, so
    a runtime timeout would leave a client polling it forever."""
    from app.services import jobs

    for status in (jobs.RUNNING, jobs.QUEUED):
        stranded = jobs.Job(
            job_id="stranded-" + status, owner="local", kind="health-report",
            dataset_id=dataset, status=status, created_at=time.time(),
            started_at=time.time() if status == jobs.RUNNING else 0.0,
            boot_id="a-process-that-is-gone")
        jobs.store.save(stranded)

    # Polling one job by id is how a client waits for a report, so the
    # answer has to be right there and not only in the list.
    for status in (jobs.RUNNING, jobs.QUEUED):
        res = client.get("/api/jobs/stranded-" + status).json()
        assert res["status"] == "failed", (status, res)
        assert "restarted" in res["error"]


def test_a_job_cannot_be_started_for_someone_elses_dataset(client):
    res = client.post("/api/jobs", json={"dataset_id": "not-a-dataset",
                                         "kind": "health-report"})
    assert res.status_code == 404


# ══════════════════════════════════════════════════════════
#  The share link — mostly what it must not do
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def shared(client, dataset):
    job = client.post("/api/jobs", json={"dataset_id": dataset,
                                         "kind": "health-report"}).json()
    done = _finish(client, job["job_id"])
    assert done["status"] == "done", done["error"]
    art = done["artifact_id"]
    link = client.post(f"/api/artifacts/{art}/share",
                       json={"ttl_days": 7, "note": "for the CFO"}).json()
    return art, link


def test_the_link_opens_with_no_account_at_all(client, shared):
    """The recipient is the CFO who does not have a login. That is the
    entire point of the feature."""
    _art, link = shared
    anonymous = TestClient(app, raise_server_exceptions=False)
    res = anonymous.get(f"/api/shared/{link['token']}")
    assert res.status_code == 200
    assert res.content[:4] == b"%PDF"


def test_the_link_uses_the_deployments_public_address(client, dataset,
                                                     monkeypatch):
    """Behind a reverse proxy the request's own base URL is the
    container's internal address, so a link built from it is dead on
    arrival for the recipient — and production is the worst place to
    discover that."""
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://analytiq.example.com/")

    job = client.post("/api/jobs", json={"dataset_id": dataset,
                                         "kind": "health-report"}).json()
    art = _finish(client, job["job_id"])["artifact_id"]
    res = client.post(f"/api/artifacts/{art}/share", json={"ttl_days": 7})
    assert res.json()["url"].startswith(
        "https://analytiq.example.com/api/shared/"), res.json()["url"]


def test_the_link_is_not_a_credential(client, shared, monkeypatch):
    """It grants exactly one artifact. If it leaks, that is the loss.

    Auth has to be switched on for this to mean anything: with no admin
    key and no accounts the app is in single-user open mode and answers
    everything, so a token — any token — would look like it worked.
    """
    from app.config import config

    _art, link = shared
    monkeypatch.setattr(config, "app_admin_key", "test-key-for-this-test")
    anonymous = TestClient(app, raise_server_exceptions=False)
    token = link["token"]

    assert anonymous.get("/api/datasets").status_code == 401, \
        "auth is not actually being enforced, so this test proves nothing"

    for path in ("/api/datasets", "/api/artifacts", "/api/jobs",
                 "/api/schedules"):
        res = anonymous.get(path, headers={"Authorization": f"Bearer {token}"})
        assert res.status_code in (401, 403), \
            "{} answered {} to a share token".format(path, res.status_code)

    # And the one thing it *is* for still works.
    assert anonymous.get(f"/api/shared/{token}").status_code == 200


@pytest.mark.parametrize("token", [
    "garbage",
    "a.b",
    "",
    "eyJvIjoiYWxpY2UifQ.0000000000000000000000000000000000000000000000000000000000000000",
    # Not ASCII. A URL path can carry anything, and this one used to
    # raise out of .encode() as a 500 on the one public route.
    "caf\u00e9.abc",
])
def test_a_forged_or_malformed_link_is_refused(client, token):
    anonymous = TestClient(app, raise_server_exceptions=False)
    assert anonymous.get(f"/api/shared/{token}").status_code == 404


def test_the_expiry_cannot_be_edited_by_the_recipient(client, shared):
    """The expiry is inside the signature, so extending it invalidates
    the token rather than extending the access."""
    import base64
    import json

    from app.services.sharing import ShareError, read

    _art, link = shared
    body, _, sig = link["token"].partition(".")
    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    payload["exp"] = time.time() + 10 ** 9
    forged = base64.urlsafe_b64encode(
        json.dumps(payload).encode()).rstrip(b"=").decode() + "." + sig

    with pytest.raises(ShareError):
        read(forged)


def test_a_link_can_be_withdrawn_before_it_expires(client, shared):
    """"We sent it to the wrong address" happens, and "wait seven days"
    is not an answer."""
    art, link = shared
    anonymous = TestClient(app, raise_server_exceptions=False)
    assert anonymous.get(f"/api/shared/{link['token']}").status_code == 200

    client.delete(f"/api/artifacts/{art}/share")
    res = anonymous.get(f"/api/shared/{link['token']}")
    assert res.status_code == 410
    assert "withdrawn" in res.json()["detail"]


def test_one_link_being_hammered_does_not_affect_another(client, dataset,
                                                       shared):
    """The share route is the only one that answers without a session,
    and every answer reads a multi-megabyte file off disk. It cannot be
    limited per account — anonymous callers are all one account, so one
    recipient refreshing would lock out every other client's CFO."""
    from app.api.deliver import SHARE_OPENS_PER_MINUTE

    _art, first = shared

    job = client.post("/api/jobs", json={"dataset_id": dataset,
                                         "kind": "health-report"}).json()
    other_art = _finish(client, job["job_id"])["artifact_id"]
    second = client.post(f"/api/artifacts/{other_art}/share",
                         json={"ttl_days": 7}).json()

    anonymous = TestClient(app, raise_server_exceptions=False)
    codes = {anonymous.get(f"/api/shared/{first['token']}").status_code
             for _ in range(SHARE_OPENS_PER_MINUTE + 5)}
    assert 429 in codes, "the public route has no ceiling at all"

    # The other client's link is untouched.
    assert anonymous.get(f"/api/shared/{second['token']}").status_code == 200


def test_following_a_link_is_recorded(client, shared):
    """Who saw the report is a question a compliance team asks, and
    "we don't log that" is the wrong answer."""
    art, link = shared
    anonymous = TestClient(app, raise_server_exceptions=False)
    anonymous.get(f"/api/shared/{link['token']}")
    anonymous.get(f"/api/shared/{link['token']}")

    meta = client.get(f"/api/artifacts/{art}").json()
    assert meta["share_views"] >= 2
    assert meta["shared"] is True


def test_granting_and_withdrawing_access_both_reach_the_audit_trail(
        client, dataset, shared):
    """Access widened and access withdrawn are both events a compliance
    team asks about. A trail that records only the grant says the report
    is still reachable long after it stopped being."""
    art, _link = shared
    client.delete(f"/api/artifacts/{art}/share")

    trail = client.get(f"/api/datasets/{dataset}/integrity").json()["audit"]
    events = [e["event"] for e in trail]
    # The build itself, not only the sharing of it. A report produced in
    # the background is the one most likely to be read months later, so
    # it is the one that most needs an entry tying it to the state of
    # the data it came from.
    assert "report" in events, events
    assert "share" in events, events
    assert "revoke" in events, events
    # Not silently filed as something else because the kind was unknown.
    assert events.count("transform") == 0, events


# ══════════════════════════════════════════════════════════
#  Arriving without being asked
# ══════════════════════════════════════════════════════════

def test_a_schedule_produces_exactly_one_report_per_window(client, dataset):
    """The tick records the PERIOD it produced, not the time it ran, so
    a restart or an overlapping tick cannot double-send — and a client
    who receives the same report twice stops trusting the ones that
    arrive."""
    client.post("/api/schedules", json={
        "dataset_id": dataset, "kind": "health-report",
        "cadence": "weekly", "hour": 0})

    first = client.post("/api/schedules/tick").json()
    assert first["started"] == 1

    second = client.post("/api/schedules/tick").json()
    assert second["started"] == 0, "the same window fired twice"


def test_a_tick_from_one_account_leaves_another_accounts_window_alone(
        client, dataset):
    """The tick endpoint is pressable by anyone signed in. If it swept
    every account, one tenant could spend another's CPU and — worse —
    claim their window, so the report that tenant is waiting for never
    fires."""
    from app.services import schedules

    theirs = schedules.create("someone-else", dataset, "health-report",
                              "weekly", hour=0)
    assert schedules.due(theirs), "this schedule should owe a report"

    client.post("/api/schedules/tick")

    after = schedules.store.get("someone-else", theirs.schedule_id)
    assert after.last_period == "", \
        "another account's tick consumed this schedule's window"
    assert schedules.due(after), "their report would now never fire"


def test_a_schedule_says_when_it_will_next_run(client, dataset):
    """A schedule whose next run you cannot see is one you do not
    trust."""
    client.post("/api/schedules", json={
        "dataset_id": dataset, "cadence": "weekly", "hour": 7})
    listed = client.get("/api/schedules").json()["schedules"][0]
    assert listed["next_run_at"] > 0
    assert listed["cadence"] == "weekly"


def test_delivery_without_email_configured_says_so(client, dataset):
    """A schedule that quietly produces no output is worse than no
    schedule. Without SMTP the report is still built and waiting, and
    the status says exactly that."""
    client.post("/api/schedules", json={
        "dataset_id": dataset, "kind": "health-report",
        "cadence": "daily", "hour": 0,
        "recipients": ["cfo@example.com"]})
    client.post("/api/schedules/tick")

    deadline = time.time() + 120
    state = {}
    while time.time() < deadline:
        state = client.get("/api/schedules").json()["schedules"][0]
        if state["last_artifact_id"]:
            break
        time.sleep(0.5)

    assert state.get("last_artifact_id"), "no report was produced"
    assert "no email configured" in state["last_status"]


def test_an_invalid_cadence_is_refused(client, dataset):
    res = client.post("/api/schedules", json={
        "dataset_id": dataset, "cadence": "fortnightly"})
    assert res.status_code == 422


# ══════════════════════════════════════════════════════════
#  Retention
# ══════════════════════════════════════════════════════════

def test_an_expired_report_stops_being_served(client, dataset, tmp_path):
    """A report is a photograph of a dataset at a moment; an old one is
    misleading rather than useful."""
    import json

    from app.services.artifacts import store as artifacts

    job = client.post("/api/jobs", json={"dataset_id": dataset,
                                         "kind": "health-report"}).json()
    done = _finish(client, job["job_id"])
    art_id = done["artifact_id"]

    path = artifacts._meta_path("local", art_id)
    with open(path) as fh:
        meta = json.load(fh)
    meta["expires_at"] = time.time() - 1
    with open(path, "w") as fh:
        json.dump(meta, fh)

    assert client.get(f"/api/artifacts/{art_id}").status_code == 404
    assert artifacts.sweep() >= 1


def test_the_sweep_does_not_delete_a_report_that_is_still_being_written(
        client):
    """put() makes the directory, writes the bytes, then writes the
    metadata. A sweep landing in that window sees a directory with no
    readable metadata — which is also what a genuinely broken artifact
    looks like — and used to delete the report someone was at that
    moment waiting for."""
    import os

    from app.services.artifacts import store as artifacts

    half_written = artifacts._dir("local", "0123456789abcdef")
    os.makedirs(half_written, exist_ok=True)
    with open(os.path.join(half_written, "blob"), "wb") as fh:
        fh.write(b"%PDF-not-finished")

    artifacts.sweep()
    assert os.path.isdir(half_written), \
        "the sweep deleted a report that was still being written"
