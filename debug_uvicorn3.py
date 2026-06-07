import sys, os, traceback

try:
    import uvicorn
    from fastapi import FastAPI
    import logging
    logging.basicConfig(level=logging.DEBUG, force=True)
    
    app = FastAPI(title="minimal")
    
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug", access_log=True)
except SystemExit as e:
    sys.stderr.write(f"SystemExit({e.code})\n")
    sys.stderr.flush()
except BaseException as e:
    sys.stderr.write(f"ERROR: {type(e).__name__}: {e}\n")
    traceback.print_exc()
    sys.stderr.flush()
