import sys, io

def enable_stdout():
    sys.stdout = sys.__stdout__


def disable_stdout():
    sys.stdout = io.StringIO()
