import sys, os, traceback

sys.stderr.write(f"Python {sys.version}\n")
sys.stderr.flush()

try:
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="minimal")
    @app.get("/")
    async def root():
        return {"ok": True}
    sys.stderr.write("FastAPI app created OK\n")
    
    port = int(os.environ.get("PORT", 8080))
    sys.stderr.write(f"Starting uvicorn on port {port}...\n")
    sys.stderr.flush()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=True)
except SystemExit as e:
    sys.stderr.write(f"SystemExit({e.code})\n")
    sys.stderr.flush()
except BaseException as e:
    sys.stderr.write(f"ERROR: {type(e).__name__}: {e}\n")
    traceback.print_exc()
    sys.stderr.flush()
