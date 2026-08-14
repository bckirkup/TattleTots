"""Interface layer: domain adapter ABC and built-in scenarios."""

from tattletots.interface.adapter_conformance import (
    AdapterConformanceCheck,
    AdapterConformanceFinding,
    AdapterConformanceReport,
    StateIndependenceFactory,
    assert_adapter_conformance,
    validate_adapter_conformance,
)
from tattletots.interface.domain_adapter import DomainAdapter
from tattletots.interface.reporter_policy import (
    ReporterDecision,
    ReporterMetadata,
    ReporterPolicy,
    ReporterPolicyContext,
    ReporterPolicyFactory,
    ReporterStream,
    create_reporter_policy,
    register_reporter_policy,
)

__all__ = [
    "DomainAdapter",
    "AdapterConformanceCheck",
    "AdapterConformanceFinding",
    "AdapterConformanceReport",
    "StateIndependenceFactory",
    "assert_adapter_conformance",
    "validate_adapter_conformance",
    "ReporterDecision",
    "ReporterMetadata",
    "ReporterPolicy",
    "ReporterPolicyContext",
    "ReporterPolicyFactory",
    "ReporterStream",
    "create_reporter_policy",
    "register_reporter_policy",
]
