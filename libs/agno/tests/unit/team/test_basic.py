from unittest.mock import AsyncMock, MagicMock

import pytest

from agno.agent import Agent
from agno.metrics import MessageMetrics, RunMetrics
from agno.models.message import Message
from agno.models.openai import OpenAIChat
from agno.run import RunContext
from agno.run.team import TeamRunOutput
from agno.session.team import TeamSession
from agno.team._run import _asetup_session
from agno.team.team import Team
from agno.tools.websearch import WebSearchTools
from agno.tools.yfinance import YFinanceTools
from agno.utils.string import is_valid_uuid


@pytest.fixture
def team():
    web_agent = Agent(
        name="Web Agent",
        model=OpenAIChat("gpt-4o"),
        role="Search the web for information",
        tools=[WebSearchTools(cache_results=True)],
    )

    finance_agent = Agent(
        name="Finance Agent",
        model=OpenAIChat("gpt-4o"),
        role="Get financial data",
        tools=[YFinanceTools(include_tools=["get_current_stock_price"])],
    )

    team = Team(name="Router Team", model=OpenAIChat("gpt-4o"), members=[web_agent, finance_agent])
    return team


def test_team_system_message_content(team):
    """Test basic functionality of a route team."""

    # Get the actual content
    members_content = team.get_members_system_message_content()

    # Check for expected content with fuzzy matching
    assert "Agent 1:" in members_content
    assert "ID: web-agent" in members_content
    assert "Name: Web Agent" in members_content
    assert "Role: Search the web for information" in members_content

    assert "Agent 2:" in members_content
    assert "ID: finance-agent" in members_content
    assert "Name: Finance Agent" in members_content
    assert "Role: Get financial data" in members_content


def test_delegate_to_wrong_member(team):
    function = team._get_delegate_task_function(
        session=TeamSession(session_id="test-session"),
        run_response=TeamRunOutput(content="Hello, world!"),
        run_context=RunContext(session_state={}, run_id="test-run", session_id="test-session"),
        team_run_context={},
    )
    response = list(function.entrypoint(member_id="wrong-agent", task="Get the current stock price of AAPL"))
    assert "Member with ID wrong-agent not found in the team or any subteams" in response[0]


def test_set_id():
    team = Team(
        id="test_id",
        members=[],
    )
    team.set_id()
    assert team.id == "test_id"


def test_set_id_from_name():
    team = Team(
        name="Test Name",
        members=[],
    )
    team.set_id()
    team_id = team.id

    assert team_id is not None
    assert team_id == "test-name"

    team.id = None
    team.set_id()
    # It is deterministic, so it should be the same
    assert team.id == team_id


def test_set_id_auto_generated():
    team = Team(
        members=[],
    )
    team.set_id()
    assert team.id is not None
    assert is_valid_uuid(team.id)


def test_team_accumulate_model_metrics(team):
    """Test that _accumulate_model_metrics accumulates metrics from model response."""
    from agno.metrics import ModelMetrics, accumulate_model_metrics
    from agno.models.response import ModelResponse

    run_response = TeamRunOutput(content="test")
    run_response.metrics = RunMetrics()
    run_response.metrics.start_timer()

    # Simulate a model response with usage metrics
    model_response = ModelResponse(content="response")
    model_response.response_usage = RunMetrics(input_tokens=10, output_tokens=20, total_tokens=30)

    team._accumulate_model_metrics(model_response, team.model, "model", run_response)

    assert run_response.metrics.input_tokens == 10
    assert run_response.metrics.output_tokens == 20
    assert run_response.metrics.total_tokens == 30
    assert run_response.metrics.details is not None
    assert "model" in run_response.metrics.details
    assert len(run_response.metrics.details["model"]) == 1


def test_team_update_session_metrics_accumulates(team):
    """Test that _update_session_metrics correctly accumulates metrics using run_response."""

    session = TeamSession(session_id="test_session")
    session.session_data = {}

    # First Run
    run1 = TeamRunOutput(content="run 1")
    run1.metrics = RunMetrics()
    run1.metrics.duration = 2.0
    run1.metrics.input_tokens = 100

    # Add run to session
    session.upsert_run(run1)
    team._update_session_metrics(session=session, run_response=run1)

    metrics1 = session.session_data["session_metrics"]
    assert metrics1["average_duration"] == 2.0
    assert metrics1["input_tokens"] == 100
    assert metrics1["total_runs"] == 1

    # Second Run
    run2 = TeamRunOutput(content="run 2")
    run2.metrics = RunMetrics()
    run2.metrics.duration = 3.0
    run2.metrics.input_tokens = 50

    # Add second run to session
    session.upsert_run(run2)
    # Should accumulate with previous session metrics
    team._update_session_metrics(session=session, run_response=run2)

    metrics2 = session.session_data["session_metrics"]

    assert metrics2["average_duration"] == 2.5  # (2.0 + 3.0) / 2
    assert metrics2["input_tokens"] == 150  # 100 + 50
    assert metrics2["total_runs"] == 2


@pytest.mark.asyncio
async def test_asetup_session_resolves_deps_after_state_loaded():
    """Verify callable dependencies are resolved AFTER session state is loaded from DB.

    This is a regression test: if dependency resolution runs before state loading,
    the callable won't see DB-stored session state values.
    """
    # Create a session with DB-stored state
    db_session = TeamSession(session_id="test-session")
    db_session.session_data = {"session_state": {"from_db": "loaded"}}

    # Track the session_state snapshot at the time _aresolve_run_dependencies is called
    captured_state = {}

    async def capture_state_on_resolve(run_context):
        """Capture session_state at dep resolution time, then do actual resolution."""
        captured_state.update(run_context.session_state or {})

    # Create a minimal Team mock
    team = MagicMock()
    team._has_async_db.return_value = False
    team._read_or_create_session.return_value = db_session
    team._update_metadata.return_value = None
    team._initialize_session_state.side_effect = lambda session_state, **kw: session_state
    team._load_session_state.side_effect = lambda session, session_state: {
        **session_state,
        **session.session_data.get("session_state", {}),
    }
    team._aresolve_run_dependencies = AsyncMock(side_effect=capture_state_on_resolve)

    run_context = RunContext(
        run_id="test-run",
        session_id="test-session",
        session_state={},
        dependencies={"some_dep": lambda: "value"},
    )

    result_session = await _asetup_session(
        team=team,
        run_context=run_context,
        session_id="test-session",
        user_id=None,
        run_id="test-run",
    )

    assert result_session == db_session
    # At the time deps were resolved, session_state should already contain DB values
    assert captured_state.get("from_db") == "loaded"
    # And run_context.session_state should have the loaded value
    assert run_context.session_state["from_db"] == "loaded"
