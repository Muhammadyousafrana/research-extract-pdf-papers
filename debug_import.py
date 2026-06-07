import sys, traceback
sys.stderr.write("Starting web_app import test...\n")
sys.stderr.flush()
try:
    import web_app
    sys.stderr.write("IMPORT OK\n")
except Exception:
    sys.stderr.write("IMPORT FAILED\n")
    traceback.print_exc()
    sys.stderr.flush()
