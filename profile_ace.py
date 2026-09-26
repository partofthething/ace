"""
Profile ACE.

Plot with a command like this:
    ../../../gprof2dot/gprof2dot.py -f pstats ace.stats  | dot -Tpng -o ace_profile.png
"""

import cProfile

import ace.ace
from ace.samples import wang04

if __name__ == "__main__":
    x, y = wang04.build_sample_ace_problem_wang04(N=200)
    ace_solver = ace.ace.ACESolver()
    ace_solver.specify_data_set(x, y)
    cProfile.run("ace_solver.solve()", "ace.stats")
