import sys, os, subprocess, time

sys.stderr.write("Testing MCP subprocess...\n")
sys.stderr.flush()

try:
    proc = subprocess.Popen(
        [sys.executable, "research_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ},
    )
    sys.stderr.write(f"PID: {proc.pid}, waiting 5s...\n")
    sys.stderr.flush()
    try:
        out, err = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    if out:
        sys.stderr.write(f"STDOUT ({len(out)} bytes): {out[:500]}\n")
    if err:
        sys.stderr.write(f"STDERR ({len(err)} bytes): {err[:500]}\n")
    sys.stderr.write(f"Exit code: {proc.returncode}\n")
except Exception as e:
    sys.stderr.write(f"Error: {e}\n")
    import traceback
    traceback.print_exc()
sys.stderr.flush()
