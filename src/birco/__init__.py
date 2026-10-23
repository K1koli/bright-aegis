"""BIRCO adapter — cross-benchmark OOD evaluation for the BRIGHT-evolved harnesses.

BIRCO differs from BRIGHT in exactly the way that makes it a useful OOD probe: the
candidate pool is GIVEN per query, so there is no retrieval step. Two of the levers
the BRIGHT runs learned therefore cannot fire (widening the pool, and gathering other
chunks of the same source document); what remains testable is the selection/ordering
behaviour itself.

The adapter is deliberately thin. ``PoolIndex`` mimics the ``BM25Index`` interface that
``bright_harness.tools.Retrieve`` already calls, so **the harness configs run unchanged**
— the same H_0 prompt, the same processors, the same scorer.
"""
