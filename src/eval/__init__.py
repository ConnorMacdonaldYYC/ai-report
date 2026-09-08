"""Evaluation framework for comparing report-generation models.

Runs the report pipeline with different generator models and judges each
report with a separate panel of evaluator models, capturing token usage
and wall-clock time for every run.
"""
