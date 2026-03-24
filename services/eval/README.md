# DermAI Eval

Phase 6 adds a lightweight local evaluation runner for the current demo state.

Run it from the repository root after the Python environment is installed:

```bash
python services/eval/run_phase6_eval.py
```

The script exercises:

- greeting handling
- grounded dermatology chat
- off-topic redirection
- emergency escalation
- image upload plus multimodal follow-up

It writes the latest result to `services/eval/artifacts/phase6-latest.json`.

For retrieval-specific checks, run:

```bash
python services/eval/run_retrieval_eval.py
```

This uses the benchmark cases in `services/eval/retrieval-benchmark.json` and writes the latest result to `services/eval/artifacts/retrieval-latest.json`.
