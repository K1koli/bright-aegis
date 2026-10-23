@echo off
REM Opens the SSH tunnel from this machine to the vLLM server on the GAAS
REM compute node. Keep this window OPEN for as long as the experiment runs;
REM closing it (or Ctrl+C) drops the tunnel and every model call will fail.
REM
REM   localhost:8000  --ssh-->  gaas.hpc.ntu.edu.sg  -->  %NODE%:8000
REM
REM If the vLLM job is resubmitted it may land on a different node -- update the
REM node name below. To find the current one:
REM     ssh gaas "bash ~/probe.sh"      (prints: node = hpc-gaas-gNN)
REM Last known: hpc-gaas-g01 (job 88302, submitted 2026-10-18 with project ds_ccds_wei.lu)
REM The 72h walltime on job 86330 expired at ~19:00 on 2026-10-18 and took the server
REM with it; the tunnel stayed up pointing at the dead g03, so every request came back
REM "ConnectionReset" instead of "ConnectionRefused". Always re-check the node after a
REM resubmit -- a live tunnel to a dead server looks like a network fault.

set NODE=hpc-gaas-g01

echo ============================================================
echo  Tunnel:  http://127.0.0.1:8000  -^>  %NODE%:8000
echo  Keep this window open. Press Ctrl+C to close the tunnel.
echo ============================================================
echo.

ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -N -L 8000:%NODE%:8000 gaas

echo.
echo Tunnel closed.
pause
