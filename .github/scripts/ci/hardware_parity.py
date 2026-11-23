# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""The required hardware and common pytest evidence for a pull request."""

import shlex


REQUIRED_PLATFORMS = frozenset(
    {
        "apple",
        "ascend",
        "enflame",
        "hygon",
        "iluvatar",
        "kunlunxin",
        "metax",
        "mthreads",
        "nvidia",
        "thead",
    }
)
GROUP_ID = "common_regression"
TEST_PATH = "tests/integration/common"
COMMAND = ("python", "-m", "pytest", TEST_PATH, "-v", "--tb=short")
REQUIRED_COMMON_CASES = frozenset(
    f"{TEST_PATH}/test_reductions.py::test_indexed_reduction[{op}]"
    for op in ("argmin", "max", "min")
)


def validate_pr_plan(plan):
    """Require the complete hardware floor and unfiltered common test command."""
    lanes = plan["execution_plan"]
    missing = REQUIRED_PLATFORMS.difference(lane["name"] for lane in lanes)
    if missing:
        raise ValueError(
            "PR plan is missing required platforms: " + ", ".join(sorted(missing))
        )
    for lane in lanes:
        groups = [group for group in lane["groups"] if group["group_id"] == GROUP_ID]
        if (
            len(groups) != 1
            or tuple(shlex.split(groups[0]["command"])) != COMMAND
            or groups[0].get("environment")
        ):
            raise ValueError(
                f"{lane['name']}: expected the unfiltered common regression group"
            )


def common_cases(report):
    """Return a lane's inventory only when the full common session passed."""
    evidence = report["evidence"]
    groups = [
        group
        for group in evidence["execution"]["groups"]
        if group["group_id"] == GROUP_ID
    ]
    if len(groups) != 1 or groups[0]["returncode"] != 0:
        raise ValueError("the common regression command did not exit successfully")
    sessions = [
        session for session in evidence["sessions"] if session["group_id"] == GROUP_ID
    ]
    if len(sessions) != 1:
        raise ValueError("expected one common regression pytest session")
    session = sessions[0]
    cases = session["selected_nodeids"]
    nonpassing = [
        f"{test['nodeid']} ({test['outcome']})"
        for test in session["tests_details"]
        if test["outcome"] != "passed"
    ] + session["collection_failures"]
    if nonpassing:
        raise ValueError("common regression did not pass: " + ", ".join(nonpassing))
    if (
        not cases
        or len(set(cases)) != len(cases)
        or any(not case.startswith(TEST_PATH + "/") for case in cases)
        or [test["nodeid"] for test in session["tests_details"]]
        != session["observed_nodeids"]
        or session["summary"].get("passed") != len(cases)
        or session["summary"].get("collected") != len(cases)
        or session["summary"].get("deselected", 0)
    ):
        raise ValueError("all common regression cases must be collected and pass")
    inventory = set(cases)
    missing = REQUIRED_COMMON_CASES - inventory
    if missing:
        raise ValueError(
            "missing required common regression cases: " + ", ".join(sorted(missing))
        )
    return inventory
