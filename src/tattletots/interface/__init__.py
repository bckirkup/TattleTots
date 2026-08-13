"""Interface layer: domain adapter ABC and built-in scenarios."""

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
    "ReporterDecision",
    "ReporterMetadata",
    "ReporterPolicy",
    "ReporterPolicyContext",
    "ReporterPolicyFactory",
    "ReporterStream",
    "create_reporter_policy",
    "register_reporter_policy",
]
