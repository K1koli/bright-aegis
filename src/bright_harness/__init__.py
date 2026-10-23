"""BRIGHT harness adapter for HarnessX.

This is the benchmark adapter layer: it turns a BRIGHT query into a
``BaseTask``, gives the task agent a ``HarnessConfig`` whose ranking behaviour is
the thing AEGIS evolves, and scores the agent's output with deterministic
nDCG@10.

Isolation contract (the anti-reward-hacking boundary):

* The harness only ever sees the query text and the corpus (doc ids + content).
* ``gold_ids`` / qrels live in the evaluator's closure and are never placed in a
  task, a tool result, a prompt, or a trajectory file.
* Scoring happens outside the harness, after ``harness.run()`` returns.
"""
