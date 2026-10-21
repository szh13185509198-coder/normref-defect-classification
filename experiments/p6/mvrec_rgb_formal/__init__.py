"""MVREC-RGB formal pipeline package (support-only infrastructure).

Contains the 195-unit execution manifest, the per-unit formal support
runner, the support-grid orchestrator/audit, the gated query evaluator,
the V1 GO/NO-GO decision evaluator, and the K8 nested manifest tooling.

All modules are support-only until the query evaluator gate is explicitly
opened by the main agent after support-grid authorization.  No module in
this package reads query performance on import.
"""
