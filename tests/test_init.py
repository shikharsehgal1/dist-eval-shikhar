"""Tests for disteval public package exports."""
import disteval


def test_public_api_exports_record_store():
    assert hasattr(disteval, "RecordStore")
    assert hasattr(disteval, "EpisodeRecord")


def test_public_api_exports_core_modules():
    assert hasattr(disteval, "metrics")
    assert hasattr(disteval, "bootstrap")
    assert hasattr(disteval, "compare")
    assert hasattr(disteval, "failure")
    assert hasattr(disteval, "right_tail")


def test_public_api_exports_self_improvement_modules():
    assert hasattr(disteval, "self_engine")
    assert hasattr(disteval, "trajectory_monitor")
    assert hasattr(disteval, "trajectory_memory")


def test_public_api_exports_recursive_modules():
    assert hasattr(disteval, "recursion_engine")
    assert hasattr(disteval, "environment_generator")
    assert hasattr(disteval, "environment_registry")
    assert hasattr(disteval, "distributed_eval")


def test_all_is_complete():
    for name in disteval.__all__:
        assert hasattr(disteval, name)
