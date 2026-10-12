#!/data/data/com.termux/files/usr/bin/bash
set -e
curl -s http://127.0.0.1:8000/health
echo
curl -s -X POST http://127.0.0.1:8000/simulations/run \
  -H "Content-Type: application/json" \
  -d "{\"system_type\":\"particle\",\"parameters\":{\"L_nm\":2.0,\"n_states\":5}}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[\"status\"], round(d[\"compute_time_ms\"],1), [round(e,6) for e in d[\"result\"][\"energies_eV\"]])"
