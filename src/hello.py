#!/usr/bin/env python3
"""
Hello World example script.
This is a simple starting point for your project.
"""

def main():
    print("Hello, World!")
    print("Testing to see if the environment is set up correctly by loading libraries ...")
    try:
        import numpy as np
    except ImportError:
        print("Failed to import numpy.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Failed to import matplotlib.")
        return
    try:
        import pandas as pd
    except ImportError:
        print("Failed to import pandas.")
        return
    try:
        import scipy
    except ImportError:
        print("Failed to import scipy.")
        return
    try:
        import sympy as sp
    except ImportError:
        print("Failed to import sympy.")
        return
    try:
        import control
    except ImportError:
        print("Failed to import control.")
        return
    print("All libraries imported successfully!")
    


if __name__ == "__main__":
    """This runs when the script is executed directly."""
    main()
