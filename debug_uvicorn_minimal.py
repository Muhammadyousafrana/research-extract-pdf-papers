import sys
sys.stderr.write(f"Python {sys.version}\n")
sys.stderr.flush()

# Test uvicorn import
try:
    import uvicorn
    sys.stderr.write(f"uvicorn imported, version: {uvicorn.__version__}\n")
except Exception as e:
    sys.stderr.write(f"uvicorn import failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.stderr.flush()

# FastAPI basic test
try:
    from fastapi import FastAPI
    app = FastAPI(title="minimal")
    @app.get("/")
    async def root():
        return {"ok": True}
    sys.stderr.write("FastAPI app created OK\n")
except Exception as e:
    sys.stderr.write(f"FastAPI create failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.stderr.flush()

# Now actually run
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
sys.stderr.write(f"Starting uvicorn on port {port}...\n")
sys.stderr.flush()
try:
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=True)
except Exception as e:
    sys.stderr.write(f"uvicorn.run failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
