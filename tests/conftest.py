"""Make "these tests are offline" a property the suite enforces, not one it hopes for.

Two of the audit tests reached doi.org for real. They passed locally, where the
call failed and the code fell through to the branch the test expected, and broke
in CI, where the call succeeded and produced the correct - but different -
answer. A test that changes its verdict depending on whether the machine has a
network is not testing anything.

Any test that shells out to curl now fails loudly with an explanation. A test
that genuinely needs the network can opt in with @pytest.mark.network.
"""
import subprocess

import pytest

_real_run = subprocess.run


def pytest_configure(config):
    config.addinivalue_line("markers", "network: test may make real network calls")


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    if request.node.get_closest_marker("network"):
        return

    def guard(cmd, *a, **kw):
        argv = cmd if isinstance(cmd, (list, tuple)) else [cmd]
        if argv and str(argv[0]).endswith("curl"):
            # pytest.fail raises a BaseException, deliberately. The network
            # helpers wrap their calls in `except Exception: return None`, so an
            # AssertionError here would be swallowed and the guard would pass
            # silently - which is what happened the first time it was written.
            pytest.fail(
                "This test made a real network call:\n"
                "    " + " ".join(str(x) for x in argv[:6]) + " ...\n"
                "Stub the function that reaches out (resolve_identifier, "
                "doi_registered, search_openalex, search_s2, _curl) so the test "
                "asserts the same thing on every machine. If it truly needs the "
                "network, mark it @pytest.mark.network.",
                pytrace=False,
            )
        return _real_run(cmd, *a, **kw)

    monkeypatch.setattr(subprocess, "run", guard)
