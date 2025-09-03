#!/bin/bash
set -Eeuo pipefail

GRACE_SECONDS=${GRACE_SECONDS:-5}

# How many parallel copies
N=12 # $(nproc)
declare -a LEADERS=()   # PIDs of job leaders (also the PGIDs we will target)

# Kill and reap helpers
kill_groups() {
  local sig="$1"; shift || true
  local p
  for p in "${LEADERS[@]}"; do
    # If the leader is still around (or its group likely exists), signal the whole group.
    if kill -0 "$p" 2>/dev/null; then
      # negative PID = process group
      kill "-$sig" "-$p" 2>/dev/null || true
    fi
  done
}

wait_all_leaders() {
  local p
  for p in "${LEADERS[@]}"; do
    # Reap direct children; this also clears zombies
    wait "$p" 2>/dev/null || true
  done
}

cleanup() {
  # Avoid reentrancy while we’re cleaning up
  trap - INT TERM

  # 1) Ask nicely
  kill_groups TERM

  # 2) Give them a moment to shut down gracefully
  local start=$SECONDS
  while :; do
    # Check if any leaders still alive
    local alive=0 p
    for p in "${LEADERS[@]}"; do
      if kill -0 "$p" 2>/dev/null; then
        alive=1; break
      fi
    done
    (( alive == 0 )) && break
    (( SECONDS - start >= GRACE_SECONDS )) && break
    sleep 0.1
  done

  # 3) If any are still up, force them
  local stubborn=()
  for p in "${LEADERS[@]}"; do
    if kill -0 "$p" 2>/dev/null; then
      stubborn+=("$p")
    fi
  done
  if ((${#stubborn[@]})); then
    echo "Forcing termination of remaining groups: ${stubborn[*]}" >&2
    kill -KILL -- "-${stubborn[@]}" 2>/dev/null || true
  fi

  # 4) Reap all children to clear zombies before exiting
  wait_all_leaders
}

on_sig() {
  echo
  echo "Caught signal — terminating process groups and waiting for shutdown..." >&2
  cleanup
  # Do not exit immediately; EXIT trap will run after main loop ends
}

trap 'on_sig' INT
trap 'on_sig' TERM
trap 'cleanup' EXIT

echo "Launching $N copies of: experiment.sh" >&2

# Launch N copies, each as a new job leader in its own process group.
# In Bash, each background job already gets its own process group whose PGID equals the job leader PID.
# We store that leader PID and later signal the negative PID to target the whole group (including grandchildren).
for ((i=0; i<N; i++)); do
  (
    # Make the child the group leader and replace its shell with the target program
    exec bash "experiment.sh" "$N" "$i"
  ) &
  LEADERS+=("$!")
done

# Keep waiting until all background children are gone.
# If your Bash supports `wait -n`, you could use that; the simple loop below is widely portable.
status=0
for pid in "${LEADERS[@]}"; do
  if ! wait "$pid"; then
    status=$?  # remember a nonzero code but keep waiting to reap everyone
  fi
done

exit "$status"
