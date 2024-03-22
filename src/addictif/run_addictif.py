#!/usr/bin/env python

import sys, importlib

list_of_scripts = ["stokes",
                   "ade_steady",
                   "refine",
                   "postprocess_abc",
                   "postprocess_crn",
                   "analyze_data",
                   "compute_averages",
                   "plot_scan",
                   "make_video"]

def main():
    assert len(sys.argv) > 1
    assert sys.argv[1] in list_of_scripts
    script = sys.argv.pop(1)
    m = importlib.import_module("addictif.scripts."+script)
    m.main()

if __name__ == "__main__":
    main()
