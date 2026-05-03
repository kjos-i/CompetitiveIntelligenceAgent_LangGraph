"""Evaluation harness for the Competitive Intelligence (CI) agent.

Marks the folder as a Python package so that imports like
from evaluation.eval_utils import ... work from the parent project,
without forcing it on the eval modules themselves (which use bare
imports such as from eval_utils import ... and continue to work
when scripts are run directly from this folder).
"""
