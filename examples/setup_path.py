import sys, os

safe_pdp = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "externals", "Safe-PDP"))
if safe_pdp not in sys.path:
    sys.path.append(safe_pdp)

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if root not in sys.path:
    sys.path.append(root)

DEMO_PATH = os.path.join(safe_pdp, "Examples", "MPC", "Demos")
