#!/bin/bash
# Validate candidates as run_persist.py produces them; exit once it has finished and nothing is left.
cd /home/adam/Documents/work/rcap-class-project/run477/validate
PY=/home/adam/Documents/work/rcap-class-project/rcap/.venv/bin/python
while true; do
  $PY validate.py fork-linkedin-kafka linkedin/kafka >> validate.log 2>&1
  if ! pgrep -f "run_persist.py" >/dev/null && grep -q "ALL DONE" persist.log; then
    $PY validate.py fork-linkedin-kafka linkedin/kafka >> validate.log 2>&1; echo "LOOP DONE" >> validate.log; break
  fi
  sleep 120
done
