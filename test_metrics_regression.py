"""
Standalone E2E regression test for the metrics system.

Tests:
  1. Import paths and backward compatibility
  2. RunMetrics / ModelMetrics / SessionModelMetrics / SessionMetrics unit behavior
  3. accumulate_model_metrics cost flow
  4. DB round-trip via real PostgreSQL (agent sessions)
  5. DB round-trip via real PostgreSQL (team sessions)
  6. Legacy format deserialization (old DB records)
  7. get_session_metrics public API returns proper types

Run:
    .venv/bin/python test_metrics_regression.py
"""

import json
import sys
import traceback
from time import time

DB_URL = "postgresql+psycopg://ai:ai@localhost:5532/ai"
TEST_SESSION_TABLE = "_test_metrics_regression_sessions"

passed = 0
failed = 0
errors = []


def test(name):
    """Decorator that runs a test function and tracks pass/fail."""
    def decorator(fn):
        global passed, failed
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            errors.append((name, str(e), tb))
            print(f"  FAIL  {name}: {e}")
        return fn
    return decorator


# ============================================================================
# 1. IMPORT PATHS
# ============================================================================
print("\n" + "=" * 70)
print("1. IMPORT PATHS")
print("=" * 70)


@test("from agno.metrics import RunMetrics")
def _():
    from agno.metrics import RunMetrics
    assert RunMetrics is not None


@test("from agno.metrics import ModelMetrics")
def _():
    from agno.metrics import ModelMetrics
    assert ModelMetrics is not None


@test("from agno.metrics import SessionModelMetrics")
def _():
    from agno.metrics import SessionModelMetrics
    assert SessionModelMetrics is not None


@test("from agno.metrics import SessionMetrics")
def _():
    from agno.metrics import SessionMetrics
    assert SessionMetrics is not None


@test("from agno.metrics import MessageMetrics")
def _():
    from agno.metrics import MessageMetrics
    assert MessageMetrics is not None


@test("from agno.metrics import ToolCallMetrics")
def _():
    from agno.metrics import ToolCallMetrics
    assert ToolCallMetrics is not None


@test("from agno.metrics import accumulate_model_metrics")
def _():
    from agno.metrics import accumulate_model_metrics
    assert accumulate_model_metrics is not None


@test("backward compat: from agno.models.metrics import Metrics")
def _():
    from agno.models.metrics import Metrics
    from agno.metrics import RunMetrics
    assert Metrics is RunMetrics


@test("backward compat: from agno.models.message import Metrics")
def _():
    from agno.models.message import Metrics
    from agno.metrics import RunMetrics
    assert Metrics is RunMetrics


# ============================================================================
# 2. RunMetrics
# ============================================================================
print("\n" + "=" * 70)
print("2. RunMetrics")
print("=" * 70)


@test("RunMetrics defaults to zero tokens")
def _():
    from agno.metrics import RunMetrics
    m = RunMetrics()
    assert m.input_tokens == 0
    assert m.output_tokens == 0
    assert m.total_tokens == 0
    assert m.cost is None


@test("RunMetrics timer start/stop sets duration")
def _():
    from agno.metrics import RunMetrics
    m = RunMetrics()
    m.start_timer()
    m.stop_timer()
    assert m.duration is not None and m.duration >= 0


@test("RunMetrics to_dict excludes zeros and timer")
def _():
    from agno.metrics import RunMetrics
    m = RunMetrics()
    d = m.to_dict()
    assert "input_tokens" not in d
    assert "timer" not in d


@test("RunMetrics to_dict includes non-zero values")
def _():
    from agno.metrics import RunMetrics
    m = RunMetrics(input_tokens=100, cost=0.01)
    d = m.to_dict()
    assert d["input_tokens"] == 100
    assert d["cost"] == 0.01


@test("RunMetrics __add__ sums tokens and cost")
def _():
    from agno.metrics import RunMetrics
    a = RunMetrics(input_tokens=10, cost=0.01)
    b = RunMetrics(input_tokens=20, cost=0.02)
    c = a + b
    assert c.input_tokens == 30
    assert abs(c.cost - 0.03) < 1e-9


@test("RunMetrics __add__ handles None cost")
def _():
    from agno.metrics import RunMetrics
    a = RunMetrics(input_tokens=10, cost=0.01)
    b = RunMetrics(input_tokens=20)
    c = a + b
    assert c.cost == 0.01


@test("RunMetrics __add__ merges provider_metrics")
def _():
    from agno.metrics import RunMetrics
    a = RunMetrics(provider_metrics={"a": 1})
    b = RunMetrics(provider_metrics={"b": 2})
    c = a + b
    assert c.provider_metrics == {"a": 1, "b": 2}


@test("RunMetrics __add__ merges additional_metrics")
def _():
    from agno.metrics import RunMetrics
    a = RunMetrics(additional_metrics={"x": 1})
    b = RunMetrics(additional_metrics={"y": 2})
    c = a + b
    assert c.additional_metrics == {"x": 1, "y": 2}


@test("RunMetrics to_dict includes details with ModelMetrics")
def _():
    from agno.metrics import RunMetrics, ModelMetrics
    mm = ModelMetrics(id="gpt-4o", provider="OpenAI", input_tokens=100, cost=0.005)
    m = RunMetrics(input_tokens=100, details={"model": [mm]})
    d = m.to_dict()
    assert "details" in d
    assert d["details"]["model"][0]["id"] == "gpt-4o"
    assert d["details"]["model"][0]["cost"] == 0.005


# ============================================================================
# 3. ModelMetrics
# ============================================================================
print("\n" + "=" * 70)
print("3. ModelMetrics")
print("=" * 70)


@test("ModelMetrics to_dict includes cost")
def _():
    from agno.metrics import ModelMetrics
    mm = ModelMetrics(id="gpt-4o", provider="OpenAI", input_tokens=100, cost=0.005)
    d = mm.to_dict()
    assert d["cost"] == 0.005
    assert d["input_tokens"] == 100


@test("ModelMetrics from_dict round-trip")
def _():
    from agno.metrics import ModelMetrics
    orig = ModelMetrics(id="gpt-4o", provider="OpenAI", input_tokens=50, cost=0.01)
    d = orig.to_dict()
    restored = ModelMetrics.from_dict(d)
    assert restored.id == "gpt-4o"
    assert restored.input_tokens == 50
    assert restored.cost == 0.01


@test("ModelMetrics from_dict ignores unknown keys")
def _():
    from agno.metrics import ModelMetrics
    d = {"id": "gpt-4o", "provider": "OpenAI", "unknown_future_field": 42}
    mm = ModelMetrics.from_dict(d)
    assert mm.id == "gpt-4o"


@test("ModelMetrics to_dict excludes zero token fields")
def _():
    from agno.metrics import ModelMetrics
    mm = ModelMetrics(id="m", provider="p", cost=0.01)
    d = mm.to_dict()
    assert "input_tokens" not in d  # zero, should be excluded
    assert "cost" in d


# ============================================================================
# 4. SessionModelMetrics
# ============================================================================
print("\n" + "=" * 70)
print("4. SessionModelMetrics")
print("=" * 70)


@test("SessionModelMetrics.from_model_metrics copies all fields")
def _():
    from agno.metrics import ModelMetrics, SessionModelMetrics
    mm = ModelMetrics(id="m", provider="p", input_tokens=10, cost=0.01)
    smm = SessionModelMetrics.from_model_metrics(mm, duration=1.5, total_runs=1)
    assert smm.input_tokens == 10
    assert smm.cost == 0.01
    assert smm.average_duration == 1.5
    assert smm.total_runs == 1
    assert smm.id == "m"


@test("SessionModelMetrics.accumulate sums tokens and cost")
def _():
    from agno.metrics import ModelMetrics, SessionModelMetrics
    smm = SessionModelMetrics(id="m", provider="p", input_tokens=10, cost=0.01, total_runs=1)
    other = ModelMetrics(id="m", provider="p", input_tokens=5, cost=0.005)
    smm.accumulate(other)
    assert smm.input_tokens == 15
    assert abs(smm.cost - 0.015) < 1e-9


@test("SessionModelMetrics.accumulate handles None cost on self")
def _():
    from agno.metrics import ModelMetrics, SessionModelMetrics
    smm = SessionModelMetrics(id="m", provider="p", input_tokens=10, total_runs=1)
    other = ModelMetrics(id="m", provider="p", input_tokens=5, cost=0.005)
    smm.accumulate(other)
    assert abs(smm.cost - 0.005) < 1e-9


@test("SessionModelMetrics.from_dict round-trip")
def _():
    from agno.metrics import SessionModelMetrics
    orig = SessionModelMetrics(id="m", provider="p", input_tokens=50, cost=0.02, total_runs=3)
    d = orig.to_dict()
    restored = SessionModelMetrics.from_dict(d)
    assert restored.input_tokens == 50
    assert restored.cost == 0.02
    assert restored.total_runs == 3


# ============================================================================
# 5. SessionMetrics
# ============================================================================
print("\n" + "=" * 70)
print("5. SessionMetrics")
print("=" * 70)


@test("SessionMetrics __add__ sums cost")
def _():
    from agno.metrics import SessionMetrics
    a = SessionMetrics(input_tokens=10, cost=0.01, total_runs=1)
    b = SessionMetrics(input_tokens=20, cost=0.02, total_runs=1)
    c = a + b
    assert abs(c.cost - 0.03) < 1e-9


@test("SessionMetrics __add__ merges provider_metrics")
def _():
    from agno.metrics import SessionMetrics
    a = SessionMetrics(total_runs=1, provider_metrics={"a": 1})
    b = SessionMetrics(total_runs=1, provider_metrics={"b": 2})
    c = a + b
    assert c.provider_metrics == {"a": 1, "b": 2}


@test("SessionMetrics __add__ merges additional_metrics")
def _():
    from agno.metrics import SessionMetrics
    a = SessionMetrics(total_runs=1, additional_metrics={"x": 1})
    b = SessionMetrics(total_runs=1, additional_metrics={"y": 2})
    c = a + b
    assert c.additional_metrics == {"x": 1, "y": 2}


@test("SessionMetrics __add__ handles None cost")
def _():
    from agno.metrics import SessionMetrics
    a = SessionMetrics(total_runs=1, cost=0.01)
    b = SessionMetrics(total_runs=1)
    c = a + b
    assert c.cost == 0.01


@test("SessionMetrics __add__ merges details by (provider, id)")
def _():
    from agno.metrics import SessionMetrics, SessionModelMetrics
    smm1 = SessionModelMetrics(id="gpt-4o", provider="OpenAI", input_tokens=10, total_runs=1)
    smm2 = SessionModelMetrics(id="gpt-4o", provider="OpenAI", input_tokens=20, total_runs=1)
    a = SessionMetrics(total_runs=1, details=[smm1])
    b = SessionMetrics(total_runs=1, details=[smm2])
    c = a + b
    assert len(c.details) == 1
    assert c.details[0].input_tokens == 30
    assert c.details[0].total_runs == 2


@test("SessionMetrics to_dict serializes details as list of dicts")
def _():
    from agno.metrics import SessionMetrics, SessionModelMetrics
    smm = SessionModelMetrics(id="gpt-4o", provider="OpenAI", input_tokens=50, total_runs=2, cost=0.01)
    sm = SessionMetrics(input_tokens=50, total_runs=2, details=[smm], cost=0.01)
    d = sm.to_dict()
    assert isinstance(d["details"], list)
    assert isinstance(d["details"][0], dict)
    assert d["details"][0]["id"] == "gpt-4o"
    assert d["cost"] == 0.01


@test("SessionMetrics to_dict -> reconstruct round-trip preserves cost/provider/additional")
def _():
    from agno.metrics import SessionMetrics, SessionModelMetrics
    smm = SessionModelMetrics(id="gpt-4o", provider="OpenAI", input_tokens=50, total_runs=2, cost=0.01)
    original = SessionMetrics(
        input_tokens=50, total_runs=2, details=[smm],
        cost=0.05, provider_metrics={"pm_key": 123}, additional_metrics={"am_key": 456}
    )
    d = original.to_dict()
    # Simulate what the DB does: JSON serialize and deserialize
    json_str = json.dumps(d)
    restored_dict = json.loads(json_str)
    # Reconstruct SessionMetrics from dict (like utils/agent.py does)
    restored_dict.pop("duration", None)
    restored_dict.pop("time_to_first_token", None)
    restored_dict.pop("timer", None)
    # Convert details
    if "details" in restored_dict and isinstance(restored_dict["details"], list):
        from agno.metrics import SessionModelMetrics as SMM
        restored_dict["details"] = [SMM.from_dict(dd) for dd in restored_dict["details"]]
    sm = SessionMetrics(**restored_dict)
    assert sm.cost == 0.05
    assert sm.provider_metrics == {"pm_key": 123}
    assert sm.additional_metrics == {"am_key": 456}
    assert len(sm.details) == 1
    assert isinstance(sm.details[0], SessionModelMetrics)
    assert sm.details[0].id == "gpt-4o"


# ============================================================================
# 6. accumulate_model_metrics
# ============================================================================
print("\n" + "=" * 70)
print("6. accumulate_model_metrics")
print("=" * 70)


@test("accumulate_model_metrics flows cost to run_response.metrics")
def _():
    from agno.metrics import RunMetrics, accumulate_model_metrics
    from agno.models.response import ModelResponse

    usage = RunMetrics(input_tokens=100, output_tokens=50, total_tokens=150, cost=0.005)
    model_response = ModelResponse(content="test")
    model_response.response_usage = usage

    class MockModel:
        id = "gpt-4o"
        def get_provider(self):
            return "OpenAI"

    class MockRunResponse:
        metrics = None

    run_response = MockRunResponse()
    accumulate_model_metrics(model_response, MockModel(), "model", run_response)
    assert run_response.metrics is not None
    assert abs(run_response.metrics.cost - 0.005) < 1e-9
    assert run_response.metrics.input_tokens == 100


@test("accumulate_model_metrics sums cost across multiple calls")
def _():
    from agno.metrics import RunMetrics, accumulate_model_metrics
    from agno.models.response import ModelResponse

    class MockModel:
        id = "gpt-4o"
        def get_provider(self):
            return "OpenAI"

    class MockRunResponse:
        metrics = None

    run_response = MockRunResponse()

    for cost in [0.01, 0.02, 0.03]:
        usage = RunMetrics(input_tokens=100, cost=cost)
        resp = ModelResponse(content="x")
        resp.response_usage = usage
        accumulate_model_metrics(resp, MockModel(), "model", run_response)

    assert abs(run_response.metrics.cost - 0.06) < 1e-9
    assert run_response.metrics.input_tokens == 300


@test("accumulate_model_metrics merges provider_metrics")
def _():
    from agno.metrics import RunMetrics, accumulate_model_metrics
    from agno.models.response import ModelResponse

    class MockModel:
        id = "gpt-4o"
        def get_provider(self):
            return "OpenAI"

    class MockRunResponse:
        metrics = None

    run_response = MockRunResponse()
    usage = RunMetrics(input_tokens=10, provider_metrics={"server_tool_use": True})
    resp = ModelResponse(content="x")
    resp.response_usage = usage
    accumulate_model_metrics(resp, MockModel(), "model", run_response)
    assert run_response.metrics.provider_metrics == {"server_tool_use": True}


# ============================================================================
# 7. MessageMetrics
# ============================================================================
print("\n" + "=" * 70)
print("7. MessageMetrics + ToolCallMetrics")
print("=" * 70)


@test("MessageMetrics from_metrics copies token fields")
def _():
    from agno.metrics import MessageMetrics, RunMetrics
    rm = RunMetrics(input_tokens=100, output_tokens=50, reasoning_tokens=20, provider_metrics={"k": 1})
    mm = MessageMetrics.from_metrics(rm)
    assert mm.input_tokens == 100
    assert mm.output_tokens == 50
    assert mm.reasoning_tokens == 20
    assert mm.provider_metrics == {"k": 1}


@test("MessageMetrics __add__ sums tokens")
def _():
    from agno.metrics import MessageMetrics
    a = MessageMetrics(input_tokens=10, output_tokens=5)
    b = MessageMetrics(input_tokens=20, output_tokens=10)
    c = a + b
    assert c.input_tokens == 30
    assert c.output_tokens == 15


@test("ToolCallMetrics timer and to_dict")
def _():
    from agno.metrics import ToolCallMetrics
    tc = ToolCallMetrics()
    tc.start_timer()
    tc.stop_timer()
    assert tc.duration is not None
    d = tc.to_dict()
    assert "timer" not in d
    assert "duration" in d


# ============================================================================
# 8. LEGACY FORMAT DESERIALIZATION
# ============================================================================
print("\n" + "=" * 70)
print("8. Legacy format deserialization")
print("=" * 70)


@test("SessionMetrics from old dict (no cost/provider_metrics/additional_metrics)")
def _():
    from agno.metrics import SessionMetrics, SessionModelMetrics
    # Old format from DB: no cost, no provider_metrics, no additional_metrics
    old_dict = {
        "input_tokens": 500,
        "output_tokens": 200,
        "total_tokens": 700,
        "total_runs": 3,
        "average_duration": 2.5,
        "details": [
            {"id": "gpt-4o", "provider": "OpenAI", "input_tokens": 500, "total_runs": 3}
        ]
    }
    # Simulate the deserialization path
    metrics_dict = old_dict.copy()
    metrics_dict.pop("duration", None)
    metrics_dict.pop("time_to_first_token", None)
    metrics_dict.pop("timer", None)
    if "details" in metrics_dict and isinstance(metrics_dict["details"], list):
        metrics_dict["details"] = [SessionModelMetrics.from_dict(d) for d in metrics_dict["details"]]
    sm = SessionMetrics(**metrics_dict)
    assert sm.input_tokens == 500
    assert sm.total_runs == 3
    assert sm.cost is None
    assert sm.provider_metrics is None
    assert sm.additional_metrics is None
    assert len(sm.details) == 1
    assert isinstance(sm.details[0], SessionModelMetrics)


@test("ModelMetrics from old dict (no cost field)")
def _():
    from agno.metrics import ModelMetrics
    old_dict = {"id": "gpt-4o", "provider": "OpenAI", "input_tokens": 100, "output_tokens": 50}
    mm = ModelMetrics.from_dict(old_dict)
    assert mm.input_tokens == 100
    assert mm.cost is None


@test("SessionModelMetrics from old dict (no cost field)")
def _():
    from agno.metrics import SessionModelMetrics
    old_dict = {"id": "gpt-4o", "provider": "OpenAI", "input_tokens": 100, "total_runs": 2}
    smm = SessionModelMetrics.from_dict(old_dict)
    assert smm.input_tokens == 100
    assert smm.cost is None
    assert smm.total_runs == 2


# ============================================================================
# 9. DB ROUND-TRIP: Agent Session
# ============================================================================
print("\n" + "=" * 70)
print("9. DB round-trip: Agent session")
print("=" * 70)


def _cleanup_test_table(db_url, table_name):
    """Drop the test table if it exists."""
    from sqlalchemy import create_engine, text
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
        conn.commit()
    engine.dispose()


@test("DB connectivity check")
def _():
    from sqlalchemy import create_engine, text
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.scalar() == 1
    engine.dispose()


@test("Agent session metrics DB round-trip preserves all fields")
def _():
    from agno.metrics import SessionMetrics, SessionModelMetrics

    table_name = TEST_SESSION_TABLE + "_agent"
    _cleanup_test_table(DB_URL, table_name)

    try:
        from agno.db.postgres import PostgresDb
        db = PostgresDb(db_url=DB_URL, session_table=table_name)

        # Build a SessionMetrics with all fields populated
        smm = SessionModelMetrics(
            id="gpt-4o", provider="OpenAI",
            input_tokens=500, output_tokens=200, total_tokens=700,
            reasoning_tokens=100, cache_read_tokens=50,
            cost=0.05, total_runs=3, average_duration=2.5
        )
        session_metrics = SessionMetrics(
            input_tokens=500, output_tokens=200, total_tokens=700,
            reasoning_tokens=100, cache_read_tokens=50,
            cost=0.05, total_runs=3, average_duration=2.5,
            details=[smm],
            provider_metrics={"accepted_prediction_tokens": 0},
            additional_metrics={"custom_key": "custom_value"},
        )

        # Write to DB
        session_data = {"session_metrics": session_metrics.to_dict()}
        from agno.session.agent import AgentSession
        from agno.db.base import SessionType
        agent_session = AgentSession(
            session_id="test_metrics_rt_agent",
            agent_id="test_agent",
            session_data=session_data,
            created_at=int(time()),
            updated_at=int(time()),
        )
        db.upsert_session(agent_session)

        # Read back from DB
        read_session = db.get_session(
            session_id="test_metrics_rt_agent",
            session_type=SessionType.AGENT,
            deserialize=False,
        )
        assert read_session is not None, "Session not found in DB"

        sm_dict = read_session.get("session_data", {}).get("session_metrics")
        assert sm_dict is not None, "session_metrics not found in DB"

        # Simulate the deserialization path (what get_session_metrics_util does)
        metrics_dict = sm_dict.copy()
        metrics_dict.pop("duration", None)
        metrics_dict.pop("time_to_first_token", None)
        metrics_dict.pop("timer", None)
        if "details" in metrics_dict and isinstance(metrics_dict["details"], list):
            metrics_dict["details"] = [SessionModelMetrics.from_dict(d) for d in metrics_dict["details"]]

        restored = SessionMetrics(**metrics_dict)

        # Verify all fields survived
        assert restored.input_tokens == 500, f"input_tokens: {restored.input_tokens}"
        assert restored.output_tokens == 200
        assert restored.total_tokens == 700
        assert restored.reasoning_tokens == 100
        assert restored.cache_read_tokens == 50
        assert restored.cost == 0.05, f"cost lost! got: {restored.cost}"
        assert restored.total_runs == 3
        assert restored.average_duration == 2.5
        assert restored.provider_metrics == {"accepted_prediction_tokens": 0}, \
            f"provider_metrics lost! got: {restored.provider_metrics}"
        assert restored.additional_metrics == {"custom_key": "custom_value"}, \
            f"additional_metrics lost! got: {restored.additional_metrics}"

        # Verify details round-trip
        assert restored.details is not None and len(restored.details) == 1
        detail = restored.details[0]
        assert isinstance(detail, SessionModelMetrics), \
            f"details[0] is {type(detail).__name__}, expected SessionModelMetrics"
        assert detail.id == "gpt-4o"
        assert detail.provider == "OpenAI"
        assert detail.input_tokens == 500
        assert detail.cost == 0.05
        assert detail.total_runs == 3

    finally:
        _cleanup_test_table(DB_URL, table_name)


@test("Agent get_session_metrics_util returns SessionModelMetrics objects (not dicts)")
def _():
    from agno.metrics import SessionMetrics, SessionModelMetrics
    from agno.utils.agent import get_session_metrics_util

    table_name = TEST_SESSION_TABLE + "_agent_util"
    _cleanup_test_table(DB_URL, table_name)

    try:
        from agno.db.postgres import PostgresDb
        from agno.agent import Agent

        db = PostgresDb(db_url=DB_URL, session_table=table_name)

        # Build session_metrics dict (as it would be stored in DB)
        smm = SessionModelMetrics(
            id="gpt-4o", provider="OpenAI",
            input_tokens=100, total_runs=1, cost=0.01
        )
        session_metrics_dict = SessionMetrics(
            input_tokens=100, total_runs=1, cost=0.01,
            details=[smm],
            provider_metrics={"k": 1},
        ).to_dict()

        # Write session to DB
        from agno.session.agent import AgentSession
        agent_session = AgentSession(
            session_id="test_util_agent",
            agent_id="test_agent_util",
            session_data={"session_metrics": session_metrics_dict},
            created_at=int(time()),
            updated_at=int(time()),
        )
        db.upsert_session(agent_session)

        # Use a minimal Agent to call get_session_metrics
        agent = Agent(name="test_agent_util", db=db, session_id="test_util_agent")
        sm = agent.get_session_metrics(session_id="test_util_agent")

        assert sm is not None, "get_session_metrics returned None"
        assert sm.cost == 0.01, f"cost lost: {sm.cost}"
        assert sm.provider_metrics == {"k": 1}, f"provider_metrics lost: {sm.provider_metrics}"
        assert sm.details is not None and len(sm.details) == 1
        detail = sm.details[0]
        assert isinstance(detail, SessionModelMetrics), \
            f"REGRESSION: details[0] is {type(detail).__name__}, expected SessionModelMetrics"
        assert detail.id == "gpt-4o"
        assert detail.cost == 0.01

    finally:
        _cleanup_test_table(DB_URL, table_name)


# ============================================================================
# 10. DB ROUND-TRIP: Team Session
# ============================================================================
print("\n" + "=" * 70)
print("10. DB round-trip: Team session")
print("=" * 70)


@test("Team get_session_metrics returns proper types from DB")
def _():
    from agno.metrics import SessionMetrics, SessionModelMetrics
    from agno.team.team import Team

    table_name = TEST_SESSION_TABLE + "_team_util"
    _cleanup_test_table(DB_URL, table_name)

    try:
        from agno.db.postgres import PostgresDb

        db = PostgresDb(db_url=DB_URL, session_table=table_name)

        smm = SessionModelMetrics(
            id="o3-mini", provider="OpenAI",
            input_tokens=200, output_tokens=50, total_tokens=250,
            cost=0.02, total_runs=2, average_duration=3.0,
        )
        session_metrics_dict = SessionMetrics(
            input_tokens=200, output_tokens=50, total_tokens=250,
            cost=0.02, total_runs=2, average_duration=3.0,
            details=[smm],
            provider_metrics={"accepted_prediction_tokens": 0},
            additional_metrics={"team_extra": "val"},
        ).to_dict()

        from agno.session.team import TeamSession
        team_session = TeamSession(
            session_id="test_team_util",
            team_id="test_team",
            session_data={"session_metrics": session_metrics_dict},
            created_at=int(time()),
            updated_at=int(time()),
        )
        db.upsert_session(team_session)

        team = Team(name="test_team", members=[], db=db, session_id="test_team_util")
        sm = team.get_session_metrics(session_id="test_team_util")

        assert sm is not None, "get_session_metrics returned None"
        assert sm.input_tokens == 200
        assert sm.cost == 0.02, f"cost lost: {sm.cost}"
        assert sm.provider_metrics == {"accepted_prediction_tokens": 0}
        assert sm.additional_metrics == {"team_extra": "val"}
        assert sm.details is not None and len(sm.details) == 1
        detail = sm.details[0]
        assert isinstance(detail, SessionModelMetrics), \
            f"REGRESSION: details[0] is {type(detail).__name__}, expected SessionModelMetrics"
        assert detail.id == "o3-mini"
        assert detail.cost == 0.02

    finally:
        _cleanup_test_table(DB_URL, table_name)


# ============================================================================
# 11. INTERNAL DESERIALIZATION (agent/_storage.py path)
# ============================================================================
print("\n" + "=" * 70)
print("11. Internal deserialization (agent/_storage.py)")
print("=" * 70)


@test("agent._storage.get_session_metrics_internal preserves cost/provider_metrics/additional_metrics")
def _():
    from agno.metrics import SessionMetrics, SessionModelMetrics
    from agno.session.agent import AgentSession

    # Simulate what the DB returns: session_data with session_metrics as a dict
    smm_dict = SessionModelMetrics(
        id="gpt-4o", provider="OpenAI", input_tokens=100, cost=0.01, total_runs=1
    ).to_dict()
    session_data = {
        "session_metrics": {
            "input_tokens": 100,
            "total_tokens": 100,
            "total_runs": 1,
            "cost": 0.01,
            "provider_metrics": {"test_pm": True},
            "additional_metrics": {"test_am": "val"},
            "details": [smm_dict],
        }
    }
    session = AgentSession(session_id="test_internal", session_data=session_data)

    # Import and call the internal function
    from agno.agent._storage import get_session_metrics_internal
    from agno.agent import Agent

    agent = Agent(name="test_internal")
    sm = get_session_metrics_internal(agent, session)

    assert sm.cost == 0.01, f"REGRESSION: cost stripped during deserialization! got: {sm.cost}"
    assert sm.provider_metrics == {"test_pm": True}, \
        f"REGRESSION: provider_metrics stripped! got: {sm.provider_metrics}"
    assert sm.additional_metrics == {"test_am": "val"}, \
        f"REGRESSION: additional_metrics stripped! got: {sm.additional_metrics}"
    assert sm.details is not None and len(sm.details) == 1
    assert isinstance(sm.details[0], SessionModelMetrics)
    assert sm.details[0].cost == 0.01


# ============================================================================
# 12. JSON SERIALIZATION STABILITY
# ============================================================================
print("\n" + "=" * 70)
print("12. JSON serialization stability")
print("=" * 70)


@test("Full metrics hierarchy survives JSON round-trip")
def _():
    from agno.metrics import (
        RunMetrics, ModelMetrics, SessionMetrics, SessionModelMetrics, MessageMetrics
    )

    # Build a complete RunMetrics with details
    mm1 = ModelMetrics(id="gpt-4o", provider="OpenAI", input_tokens=100, cost=0.01,
                       provider_metrics={"pm": 1})
    mm2 = ModelMetrics(id="claude", provider="Anthropic", input_tokens=200, cost=0.02,
                       reasoning_tokens=50)
    rm = RunMetrics(
        input_tokens=300, output_tokens=100, total_tokens=400,
        reasoning_tokens=50, cost=0.03, duration=5.0,
        details={"model": [mm1], "reasoning_model": [mm2]},
        provider_metrics={"pm_run": True},
        additional_metrics={"am_run": 42},
    )

    # Serialize
    d = rm.to_dict()
    json_str = json.dumps(d)

    # Deserialize
    restored_dict = json.loads(json_str)
    assert restored_dict["input_tokens"] == 300
    assert restored_dict["cost"] == 0.03
    assert restored_dict["provider_metrics"] == {"pm_run": True}
    assert restored_dict["additional_metrics"] == {"am_run": 42}
    assert len(restored_dict["details"]["model"]) == 1
    assert restored_dict["details"]["model"][0]["cost"] == 0.01
    assert restored_dict["details"]["reasoning_model"][0]["reasoning_tokens"] == 50


@test("SessionMetrics to_dict -> JSON -> reconstruct is lossless")
def _():
    from agno.metrics import SessionMetrics, SessionModelMetrics

    smm1 = SessionModelMetrics(id="gpt-4o", provider="OpenAI", input_tokens=100,
                                cost=0.01, total_runs=2, average_duration=1.5)
    smm2 = SessionModelMetrics(id="claude", provider="Anthropic", input_tokens=200,
                                cost=0.02, total_runs=3, reasoning_tokens=50)

    original = SessionMetrics(
        input_tokens=300, output_tokens=100, total_tokens=400,
        reasoning_tokens=50, cost=0.03, total_runs=5, average_duration=2.0,
        details=[smm1, smm2],
        provider_metrics={"pm": True},
        additional_metrics={"am": 42},
    )

    # Full round-trip
    json_str = json.dumps(original.to_dict())
    d = json.loads(json_str)

    # Reconstruct (simulating get_session_metrics_util)
    d.pop("duration", None)
    d.pop("time_to_first_token", None)
    d.pop("timer", None)
    if "details" in d and isinstance(d["details"], list):
        d["details"] = [SessionModelMetrics.from_dict(dd) for dd in d["details"]]
    restored = SessionMetrics(**d)

    assert restored.input_tokens == 300
    assert restored.cost == 0.03
    assert restored.total_runs == 5
    assert restored.average_duration == 2.0
    assert restored.provider_metrics == {"pm": True}
    assert restored.additional_metrics == {"am": 42}
    assert len(restored.details) == 2
    assert all(isinstance(x, SessionModelMetrics) for x in restored.details)
    assert restored.details[0].cost == 0.01
    assert restored.details[1].cost == 0.02


# ============================================================================
# 13. MULTI-RUN ACCUMULATION
# ============================================================================
print("\n" + "=" * 70)
print("13. Multi-run accumulation simulation")
print("=" * 70)


@test("Simulate 3 runs accumulating into session metrics")
def _():
    """Simulate what agent/_storage.update_session_metrics does across 3 runs."""
    from agno.metrics import SessionMetrics, SessionModelMetrics, RunMetrics, ModelMetrics

    session_metrics = SessionMetrics()

    for i in range(3):
        run_cost = 0.01 * (i + 1)
        mm = ModelMetrics(
            id="gpt-4o", provider="OpenAI",
            input_tokens=100 * (i + 1), output_tokens=50, total_tokens=150 * (i + 1),
            cost=run_cost,
        )
        run_metrics = RunMetrics(
            input_tokens=100 * (i + 1), output_tokens=50, total_tokens=150 * (i + 1),
            cost=run_cost, duration=1.0 + i,
            details={"model": [mm]},
            provider_metrics={f"run_{i}": True},
        )

        # Accumulate (mimic agent/_storage.update_session_metrics)
        session_metrics.input_tokens += run_metrics.input_tokens
        session_metrics.output_tokens += run_metrics.output_tokens
        session_metrics.total_tokens += run_metrics.total_tokens

        if run_metrics.cost is not None:
            session_metrics.cost = (session_metrics.cost or 0) + run_metrics.cost

        if run_metrics.provider_metrics is not None:
            if session_metrics.provider_metrics is None:
                session_metrics.provider_metrics = {}
            session_metrics.provider_metrics.update(run_metrics.provider_metrics)

        session_metrics.total_runs += 1
        if run_metrics.duration is not None:
            if session_metrics.average_duration is None:
                session_metrics.average_duration = run_metrics.duration
            else:
                total_dur = session_metrics.average_duration * (session_metrics.total_runs - 1) + run_metrics.duration
                session_metrics.average_duration = total_dur / session_metrics.total_runs

        # Per-model details
        if run_metrics.details:
            if session_metrics.details is None:
                session_metrics.details = []
            details_dict = {(m.provider, m.id): m for m in session_metrics.details}
            for model_type, model_metrics_list in run_metrics.details.items():
                for model_metrics in model_metrics_list:
                    key = (model_metrics.provider, model_metrics.id)
                    if key not in details_dict:
                        details_dict[key] = SessionModelMetrics.from_model_metrics(
                            model_metrics, duration=run_metrics.duration, total_runs=1
                        )
                    else:
                        existing = details_dict[key]
                        existing.accumulate(model_metrics)
                        existing.total_runs += 1
            session_metrics.details = list(details_dict.values())

    # Verify totals
    assert session_metrics.input_tokens == 600  # 100 + 200 + 300
    assert session_metrics.total_runs == 3
    assert abs(session_metrics.cost - 0.06) < 1e-9  # 0.01 + 0.02 + 0.03
    assert session_metrics.provider_metrics == {"run_0": True, "run_1": True, "run_2": True}
    assert len(session_metrics.details) == 1
    assert session_metrics.details[0].input_tokens == 600
    assert session_metrics.details[0].total_runs == 3
    assert abs(session_metrics.details[0].cost - 0.06) < 1e-9

    # Now test DB round-trip of the accumulated metrics
    d = session_metrics.to_dict()
    json_str = json.dumps(d)
    restored_dict = json.loads(json_str)
    restored_dict.pop("duration", None)
    restored_dict.pop("time_to_first_token", None)
    restored_dict.pop("timer", None)
    if "details" in restored_dict:
        restored_dict["details"] = [SessionModelMetrics.from_dict(dd) for dd in restored_dict["details"]]
    restored = SessionMetrics(**restored_dict)

    assert restored.input_tokens == 600
    assert abs(restored.cost - 0.06) < 1e-9
    assert restored.total_runs == 3
    assert len(restored.details) == 1
    assert abs(restored.details[0].cost - 0.06) < 1e-9


@test("Multi-run DB round-trip via real PostgreSQL")
def _():
    from agno.metrics import SessionMetrics, SessionModelMetrics

    table_name = TEST_SESSION_TABLE + "_multirun"
    _cleanup_test_table(DB_URL, table_name)

    try:
        from agno.db.postgres import PostgresDb

        db = PostgresDb(db_url=DB_URL, session_table=table_name)

        # Build accumulated session_metrics (as if 3 runs completed)
        smm = SessionModelMetrics(
            id="gpt-4o", provider="OpenAI",
            input_tokens=600, output_tokens=150, total_tokens=750,
            cost=0.06, total_runs=3, average_duration=2.0,
        )
        sm = SessionMetrics(
            input_tokens=600, output_tokens=150, total_tokens=750,
            cost=0.06, total_runs=3, average_duration=2.0,
            details=[smm],
            provider_metrics={"run_0": True, "run_1": True, "run_2": True},
            additional_metrics={"custom": "data"},
        )

        # Write
        from agno.session.agent import AgentSession
        agent_session = AgentSession(
            session_id="test_multirun",
            agent_id="test_agent",
            session_data={"session_metrics": sm.to_dict()},
            created_at=int(time()),
            updated_at=int(time()),
        )
        db.upsert_session(agent_session)

        # Read back via Agent.get_session_metrics
        from agno.agent import Agent
        agent = Agent(name="test_agent", db=db, session_id="test_multirun")
        restored = agent.get_session_metrics(session_id="test_multirun")

        assert restored is not None
        assert restored.input_tokens == 600
        assert abs(restored.cost - 0.06) < 1e-9, f"cost: {restored.cost}"
        assert restored.total_runs == 3
        assert restored.provider_metrics == {"run_0": True, "run_1": True, "run_2": True}
        assert restored.additional_metrics == {"custom": "data"}
        assert len(restored.details) == 1
        assert isinstance(restored.details[0], SessionModelMetrics)
        assert abs(restored.details[0].cost - 0.06) < 1e-9

    finally:
        _cleanup_test_table(DB_URL, table_name)


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Total:  {passed + failed}")

if errors:
    print("\nFailed tests:")
    for name, err, tb in errors:
        print(f"\n  {name}")
        print(f"    Error: {err}")
        # Show last 3 lines of traceback for context
        tb_lines = tb.strip().split("\n")
        for line in tb_lines[-3:]:
            print(f"    {line}")

print("=" * 70)
sys.exit(0 if failed == 0 else 1)
