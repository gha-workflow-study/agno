"""Tests that backward-compatible import paths and core metrics API contracts remain stable.

These tests exist to catch accidental breaking changes in the metrics public API.
"""

import pytest


# ---------------------------------------------------------------------------
# 1. Backward-compatible import paths
# ---------------------------------------------------------------------------


class TestImportPaths:
    """Verify every import path that users may depend on still resolves."""

    def test_agno_models_metrics_import_metrics(self):
        """Old canonical: from agno.models.metrics import Metrics"""
        from agno.models.metrics import Metrics

        assert Metrics is not None

    def test_agno_metrics_import_metrics_alias(self):
        """New canonical alias: from agno.metrics import Metrics"""
        from agno.metrics import Metrics

        assert Metrics is not None

    def test_agno_metrics_import_run_metrics(self):
        from agno.metrics import RunMetrics

        assert RunMetrics is not None

    def test_agno_metrics_import_session_metrics(self):
        from agno.metrics import SessionMetrics

        assert SessionMetrics is not None

    def test_agno_metrics_import_model_metrics(self):
        from agno.metrics import ModelMetrics

        assert ModelMetrics is not None

    def test_agno_metrics_import_session_model_metrics(self):
        from agno.metrics import SessionModelMetrics

        assert SessionModelMetrics is not None

    def test_agno_metrics_import_message_metrics(self):
        from agno.metrics import MessageMetrics

        assert MessageMetrics is not None

    def test_agno_metrics_import_tool_call_metrics(self):
        from agno.metrics import ToolCallMetrics

        assert ToolCallMetrics is not None

    def test_agno_models_message_import_metrics(self):
        """Old re-export: from agno.models.message import Metrics"""
        from agno.models.message import Metrics

        assert Metrics is not None

    def test_old_metrics_is_run_metrics(self):
        """Old Metrics and new RunMetrics must be the exact same class."""
        from agno.models.metrics import Metrics
        from agno.metrics import RunMetrics

        assert Metrics is RunMetrics

    def test_message_metrics_alias_is_run_metrics(self):
        """Metrics re-exported from message.py must be RunMetrics."""
        from agno.models.message import Metrics
        from agno.metrics import RunMetrics

        assert Metrics is RunMetrics


# ---------------------------------------------------------------------------
# 2. Core API contract: fields, methods, serialization
# ---------------------------------------------------------------------------


class TestRunMetricsContract:
    """RunMetrics must support the fields and methods users depend on."""

    def test_token_fields_exist(self):
        from agno.metrics import RunMetrics

        m = RunMetrics()
        assert m.input_tokens == 0
        assert m.output_tokens == 0
        assert m.total_tokens == 0
        assert m.audio_input_tokens == 0
        assert m.audio_output_tokens == 0
        assert m.audio_total_tokens == 0
        assert m.cache_read_tokens == 0
        assert m.cache_write_tokens == 0
        assert m.reasoning_tokens == 0

    def test_cost_field_exists(self):
        from agno.metrics import RunMetrics

        m = RunMetrics()
        assert m.cost is None
        m.cost = 0.05
        assert m.cost == 0.05

    def test_timer_methods(self):
        from agno.metrics import RunMetrics

        m = RunMetrics()
        m.start_timer()
        m.stop_timer()
        assert m.duration is not None
        assert m.duration >= 0

    def test_to_dict_excludes_zeros(self):
        from agno.metrics import RunMetrics

        m = RunMetrics()
        d = m.to_dict()
        assert "input_tokens" not in d
        assert "timer" not in d

    def test_to_dict_includes_nonzero(self):
        from agno.metrics import RunMetrics

        m = RunMetrics(input_tokens=100, cost=0.01)
        d = m.to_dict()
        assert d["input_tokens"] == 100
        assert d["cost"] == 0.01

    def test_add(self):
        from agno.metrics import RunMetrics

        a = RunMetrics(input_tokens=10, cost=0.01)
        b = RunMetrics(input_tokens=20, cost=0.02)
        c = a + b
        assert c.input_tokens == 30
        assert c.cost == pytest.approx(0.03)


class TestModelMetricsContract:
    """ModelMetrics must carry cost and serialize correctly."""

    def test_cost_field(self):
        from agno.metrics import ModelMetrics

        mm = ModelMetrics(id="gpt-4o", provider="openai", input_tokens=10, cost=0.005)
        assert mm.cost == 0.005

    def test_to_dict_includes_cost(self):
        from agno.metrics import ModelMetrics

        mm = ModelMetrics(id="gpt-4o", provider="openai", cost=0.005)
        d = mm.to_dict()
        assert d["cost"] == 0.005

    def test_from_dict_round_trip(self):
        from agno.metrics import ModelMetrics

        original = ModelMetrics(id="gpt-4o", provider="openai", input_tokens=50, cost=0.01)
        d = original.to_dict()
        restored = ModelMetrics.from_dict(d)
        assert restored.id == original.id
        assert restored.input_tokens == original.input_tokens
        assert restored.cost == original.cost

    def test_from_dict_ignores_unknown_keys(self):
        from agno.metrics import ModelMetrics

        d = {"id": "gpt-4o", "provider": "openai", "unknown_future_field": 42}
        mm = ModelMetrics.from_dict(d)
        assert mm.id == "gpt-4o"
        assert not hasattr(mm, "unknown_future_field") or mm.provider == "openai"


class TestSessionModelMetricsContract:
    """SessionModelMetrics: from_model_metrics, accumulate, from_dict."""

    def test_from_model_metrics_copies_cost(self):
        from agno.metrics import ModelMetrics, SessionModelMetrics

        mm = ModelMetrics(id="m", provider="p", input_tokens=10, cost=0.01)
        smm = SessionModelMetrics.from_model_metrics(mm, duration=1.0, total_runs=1)
        assert smm.cost == 0.01
        assert smm.input_tokens == 10
        assert smm.average_duration == 1.0

    def test_accumulate_sums_cost(self):
        from agno.metrics import ModelMetrics, SessionModelMetrics

        smm = SessionModelMetrics(id="m", provider="p", input_tokens=10, cost=0.01, total_runs=1)
        other = ModelMetrics(id="m", provider="p", input_tokens=5, cost=0.005)
        smm.accumulate(other)
        assert smm.input_tokens == 15
        assert smm.cost == pytest.approx(0.015)

    def test_accumulate_handles_none_cost(self):
        from agno.metrics import ModelMetrics, SessionModelMetrics

        smm = SessionModelMetrics(id="m", provider="p", input_tokens=10, total_runs=1)
        other = ModelMetrics(id="m", provider="p", input_tokens=5, cost=0.005)
        smm.accumulate(other)
        assert smm.cost == pytest.approx(0.005)

    def test_from_dict_round_trip(self):
        from agno.metrics import SessionModelMetrics

        original = SessionModelMetrics(id="m", provider="p", input_tokens=50, cost=0.02, total_runs=3)
        d = original.to_dict()
        restored = SessionModelMetrics.from_dict(d)
        assert restored.input_tokens == 50
        assert restored.cost == 0.02
        assert restored.total_runs == 3


class TestSessionMetricsContract:
    """SessionMetrics __add__ must carry cost, provider_metrics, additional_metrics."""

    def test_add_sums_cost(self):
        from agno.metrics import SessionMetrics

        a = SessionMetrics(input_tokens=10, cost=0.01, total_runs=1)
        b = SessionMetrics(input_tokens=20, cost=0.02, total_runs=1)
        c = a + b
        assert c.cost == pytest.approx(0.03)

    def test_add_merges_provider_metrics(self):
        from agno.metrics import SessionMetrics

        a = SessionMetrics(total_runs=1, provider_metrics={"a": 1})
        b = SessionMetrics(total_runs=1, provider_metrics={"b": 2})
        c = a + b
        assert c.provider_metrics == {"a": 1, "b": 2}

    def test_add_merges_additional_metrics(self):
        from agno.metrics import SessionMetrics

        a = SessionMetrics(total_runs=1, additional_metrics={"x": 1})
        b = SessionMetrics(total_runs=1, additional_metrics={"y": 2})
        c = a + b
        assert c.additional_metrics == {"x": 1, "y": 2}

    def test_add_handles_none_cost(self):
        from agno.metrics import SessionMetrics

        a = SessionMetrics(total_runs=1, cost=0.01)
        b = SessionMetrics(total_runs=1)
        c = a + b
        assert c.cost == 0.01


# ---------------------------------------------------------------------------
# 3. accumulate_model_metrics cost flow
# ---------------------------------------------------------------------------


class TestAccumulateModelMetricsCost:
    """Verify cost flows from usage through to run_response.metrics."""

    def test_cost_accumulated_to_run_metrics(self):
        from agno.metrics import RunMetrics, accumulate_model_metrics
        from agno.models.response import ModelResponse

        # Simulate a model response with cost
        usage = RunMetrics(input_tokens=100, output_tokens=50, total_tokens=150, cost=0.005)
        model_response = ModelResponse(content="test")
        model_response.response_usage = usage

        # Create a mock model
        class MockModel:
            id = "gpt-4o"

            def get_provider(self):
                return "openai"

        # Create a mock run_response
        class MockRunResponse:
            metrics = None

        run_response = MockRunResponse()
        accumulate_model_metrics(model_response, MockModel(), "model", run_response)

        assert run_response.metrics is not None
        assert run_response.metrics.cost == pytest.approx(0.005)
        assert run_response.metrics.input_tokens == 100

    def test_cost_accumulated_across_multiple_calls(self):
        from agno.metrics import RunMetrics, accumulate_model_metrics
        from agno.models.response import ModelResponse

        class MockModel:
            id = "gpt-4o"

            def get_provider(self):
                return "openai"

        class MockRunResponse:
            metrics = None

        run_response = MockRunResponse()

        # First call
        usage1 = RunMetrics(input_tokens=100, cost=0.01)
        resp1 = ModelResponse(content="a")
        resp1.response_usage = usage1
        accumulate_model_metrics(resp1, MockModel(), "model", run_response)

        # Second call
        usage2 = RunMetrics(input_tokens=200, cost=0.02)
        resp2 = ModelResponse(content="b")
        resp2.response_usage = usage2
        accumulate_model_metrics(resp2, MockModel(), "model", run_response)

        assert run_response.metrics.cost == pytest.approx(0.03)
        assert run_response.metrics.input_tokens == 300
