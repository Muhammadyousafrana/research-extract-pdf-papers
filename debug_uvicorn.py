import sys, os, traceback

sys.stderr.write("Starting uvicorn standalone...\n")
sys.stderr.flush()
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    app = FastAPI(title="Test")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    
    @app.get("/")
    async def root():
        return {"status": "ok"}
    
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
except SystemExit:
    sys.stderr.write("Caught SystemExit\n")
    sys.stderr.flush()
    raise
except BaseException:
    sys.stderr.write("Caught BaseException\n")
    traceback.print_exc()
    sys.stderr.flush()
    raise
