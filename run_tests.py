import contextlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import src.strategy as strat

with open('out_clean.txt', 'w', encoding='utf-8') as f:
    with contextlib.redirect_stdout(f):
        strat.test_strategy()
