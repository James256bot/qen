#!/data/data/com.termux/files/usr/bin/bash
SYS="$1"
PARAMS="${2:-{\}}"

curl -s -X POST http://127.0.0.1:8000/simulations/run \
  -H "Content-Type: application/json" \
  -d "{\"system_type\":\"$SYS\",\"parameters\":$PARAMS}" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
if d.get(\"status\") != \"completed\":
    print(\"ERROR:\", d)
    sys.exit(1)
r = d[\"result\"]
print(f\"status={d[\"status\"]}  time_ms={d[\"compute_time_ms\"]:.2f}\")
if \"levels\" in r:
    for L in r[\"levels\"][:12]:
        if \"n\" in L:
            print(f\"  n={L[\"n\"]}  E={L[\"energy_eV\"]:.4f} eV  deg={L.get(\"degeneracy\",1)}\")
        else:
            qs = \" | \".join(
                f\"({q.get(\"nx\",\"?\")},{q.get(\"ny\",\"?\")}\" +
                (f\",{q[\"nz\"]}\" if \"nz\" in q else \"\") + \")\"
                for q in L.get(\"quanta\", [])
            )
            print(f\"  E={L[\"energy_eV\"]:.4f} eV  deg={L[\"degeneracy\"]}  {qs}\")
elif \"energies_eV\" in r:
    for i, E in enumerate(r[\"energies_eV\"][:10], 1):
        print(f\"  state {i}: E={E:.4f} eV\")
elif \"frames\" in r:
    print(f\"  {len(r[\"frames\"])} frames  total_prob={r[\"total_prob_from_expansion\"]:.4f}\")
else:
    print(\"  keys:\", list(r.keys()))
"
