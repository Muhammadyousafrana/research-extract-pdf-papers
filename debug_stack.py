import sys, os, traceback

_real_exit = sys.exit
def _debug_exit(code=None):
    sys.stderr.write(f"sys.exit({code}) called from:\n")
    traceback.print_stack()
    sys.stderr.flush()
    _real_exit(code)
sys.exit = _debug_exit

try:
    import uvicorn
    from fastapi import FastAPI
    app = FastAPI(title="minimal")
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
except BaseException as e:
    sys.stderr.write(f"Exception: {type(e).__name__}: {e}\n")
    traceback.print_exc()
    sys.stderr.flush()
