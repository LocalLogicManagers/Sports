import json, sys, os
sys.path.insert(0, "/home/claude/sportspredict")
from engine import field_probabilities, race_pick

DATA = "/home/claude/sportspredict/data"

with open(os.path.join(DATA, "horse_racing_races.json")) as f:
    races = json.load(f)

out = []
for race in races:
    field, overround = field_probabilities(race["field"])
    field.sort(key=lambda h: -h["norm_prob"])
    pick = race_pick(field)
    out.append({
        **{k: v for k, v in race.items() if k != "field"},
        "field": field,
        "overround": overround,
        **pick,
    })

with open(os.path.join(DATA, "horse_predictions.json"), "w") as f:
    json.dump({"races": out}, f, indent=2)

for r in out:
    print(r["race"], r["date"], "-> pick:", r["pick"], f"({r['pick_prob']*100:.1f}%)", r["confidence"])
    for h in r["field"]:
        print(f"   {h['post']:>2} {h['horse']:20s} ml={h['ml_odds']:6s} norm_prob={h['norm_prob']:.3f}")
