import csv, math, json, os, sys
sys.path.insert(0, "/home/claude/sportspredict")
from engine import win_prob, shrink, make_pick

DATA = "/home/claude/sportspredict/data"

def read_csv(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))

with open(os.path.join(DATA, "model_params.json")) as f:
    PARAMS = json.load(f)

def P(league):
    return PARAMS["leagues"][league]

def build_game(league, away, home, date, rh, ra, note=None):
    p = P(league)
    home_prob = win_prob(rh, ra, p["home_adv"], p["sigma"])
    g = {"away": away, "home": home, "date": date, "home_win_prob": round(home_prob, 3)}
    if note:
        g["note"] = note
    g.update(make_pick(away, home, home_prob))
    return g

results = {"generated_at": "2026-08-29", "model_version": PARAMS["version"], "leagues": {}}

# ---------------- NFL ----------------
nfl_rows = read_csv("nfl_2025_final.csv")
nfl_ratings = {}
for r in nfl_rows:
    w, l, t = int(r["w"]), int(r["l"]), int(r["t"])
    gp = w + l + t
    pd_per_game = int(r["point_diff"]) / gp
    nfl_ratings[r["team"]] = round(pd_per_game * 0.70, 3)  # preseason carryover regression

nfl_week1 = [
    ("New England Patriots", "Seattle Seahawks", "2026-09-09"),
    ("San Francisco 49ers", "Los Angeles Rams", "2026-09-10"),
    ("Chicago Bears", "Carolina Panthers", "2026-09-13"),
    ("Tampa Bay Buccaneers", "Cincinnati Bengals", "2026-09-13"),
    ("New Orleans Saints", "Detroit Lions", "2026-09-13"),
    ("Buffalo Bills", "Houston Texans", "2026-09-13"),
    ("Baltimore Ravens", "Indianapolis Colts", "2026-09-13"),
    ("Cleveland Browns", "Jacksonville Jaguars", "2026-09-13"),
    ("Atlanta Falcons", "Pittsburgh Steelers", "2026-09-13"),
    ("New York Jets", "Tennessee Titans", "2026-09-13"),
    ("Arizona Cardinals", "Los Angeles Chargers", "2026-09-13"),
    ("Miami Dolphins", "Las Vegas Raiders", "2026-09-13"),
    ("Green Bay Packers", "Minnesota Vikings", "2026-09-13"),
    ("Washington Commanders", "Philadelphia Eagles", "2026-09-13"),
    ("Dallas Cowboys", "New York Giants", "2026-09-13"),
    ("Denver Broncos", "Kansas City Chiefs", "2026-09-14"),
]
nfl_games = [build_game("nfl", a, h, d, nfl_ratings.get(h, 0), nfl_ratings.get(a, 0)) for a, h, d in nfl_week1]
results["leagues"]["nfl"] = {
    "label": "NFL",
    "status": "preseason — ratings carried over (regressed) from final 2025 standings",
    "ratings": nfl_ratings, "games": nfl_games,
}

# ---------------- MLB ----------------
mlb_rows = read_csv("mlb_2026_standings.csv")
mlb_ratings = {}
for r in mlb_rows:
    w, l = int(r["w"]), int(r["l"])
    gp = w + l
    raw = int(r["run_diff"]) / gp
    mlb_ratings[r["team"]] = round(shrink(raw, gp, P("mlb")["shrink_k"]), 3)

mlb_upcoming = [
    ("Miami Marlins", "Washington Nationals", "2026-08-30"),
    ("Boston Red Sox", "New York Yankees", "2026-08-30"),
    ("Colorado Rockies", "Atlanta Braves", "2026-08-30"),
    ("Seattle Mariners", "Toronto Blue Jays", "2026-08-30"),
    ("Kansas City Royals", "Cleveland Guardians", "2026-08-30"),
    ("Los Angeles Dodgers", "Detroit Tigers", "2026-08-30"),
    ("San Diego Padres", "Tampa Bay Rays", "2026-08-30"),
]
mlb_games = [build_game("mlb", a, h, d, mlb_ratings.get(h, 0), mlb_ratings.get(a, 0)) for a, h, d in mlb_upcoming]
results["leagues"]["mlb"] = {"label": "MLB", "status": "in season", "ratings": mlb_ratings, "games": mlb_games}

# ---------------- Premier League ----------------
epl_rows = read_csv("epl_2026_27_standings.csv")
epl_ratings = {}
for r in epl_rows:
    gp = int(r["gp"])
    raw = int(r["gd"]) / gp if gp else 0
    epl_ratings[r["team"]] = round(shrink(raw, gp, P("epl")["shrink_k"]), 3)

epl_upcoming = [
    ("Nottingham Forest", "Liverpool", "2026-08-29"),
    ("Everton", "AFC Bournemouth", "2026-08-29"),
    ("Hull City", "Coventry City", "2026-08-29"),
    ("Newcastle United", "Tottenham Hotspur", "2026-08-29"),
    ("Brighton and Hove Albion", "Chelsea", "2026-08-30"),
    ("Brentford", "Leeds United", "2026-08-30"),
    ("Fulham", "Sunderland", "2026-08-30"),
]
epl_games = [build_game("epl", a, h, d, epl_ratings.get(h, 0), epl_ratings.get(a, 0)) for a, h, d in epl_upcoming]
results["leagues"]["epl"] = {
    "label": "Premier League",
    "status": "in season (matchweek 2) — early-season ratings are heavily shrunk toward league average",
    "ratings": epl_ratings, "games": epl_games,
}

# ---------------- MLS ----------------
mls_rows = read_csv("mls_2026_standings.csv")
mls_ratings = {}
for r in mls_rows:
    gp = int(r["gp"])
    raw = int(r["gd"]) / gp if gp else 0
    mls_ratings[r["team"]] = round(shrink(raw, gp, P("mls")["shrink_k"]), 3)

mls_upcoming = [
    ("Chicago Fire FC", "Seattle Sounders FC", "2026-08-29"),
    ("New England Revolution", "Columbus Crew", "2026-08-29"),
    ("Charlotte FC", "Atlanta United FC", "2026-08-29"),
    ("LAFC", "D.C. United", "2026-08-29"),
    ("CF Montreal", "Inter Miami CF", "2026-08-29"),
    ("Philadelphia Union", "Red Bull New York", "2026-08-29"),
    ("New York City FC", "Toronto FC", "2026-08-29"),
]
mls_games = [build_game("mls", a, h, d, mls_ratings.get(h, 0), mls_ratings.get(a, 0)) for a, h, d in mls_upcoming]
results["leagues"]["mls"] = {"label": "MLS", "status": "in season", "ratings": mls_ratings, "games": mls_games}

# ---------------- PLL Lacrosse ----------------
pll_rows = read_csv("pll_2026_standings.csv")
pll_ratings = {}
for r in pll_rows:
    w, l = int(r["w"]), int(r["l"])
    gp = w + l
    win_pct = w / gp if gp else 0.5
    pll_ratings[r["team"]] = round((win_pct - 0.5) * 10, 3)

pll_playoffs = [
    ("California Redwoods", "Denver Outlaws", "2026-08-29", "Quarterfinal"),
    ("Boston Cannons", "Maryland Whipsnakes", "2026-08-29", "Quarterfinal"),
]
pll_games = [build_game("pll", a, h, d, pll_ratings.get(h, 0), pll_ratings.get(a, 0), note) for a, h, d, note in pll_playoffs]
results["leagues"]["pll"] = {
    "label": "PLL Lacrosse",
    "status": "playoffs (quarterfinals) — neutral site, no home advantage applied",
    "ratings": pll_ratings, "games": pll_games,
}

# ---------------- NCAA Football (AP rank proxy) ----------------
ap25 = ["Ohio State","Oregon","Georgia","Notre Dame","Texas","Indiana","Miami","Texas A&M",
        "Ole Miss","Oklahoma","LSU","Texas Tech","Alabama","BYU","USC","Michigan","Washington",
        "Penn State","SMU","Tennessee","Utah","Iowa","Houston","Louisville","Missouri"]
ncaaf_ratings = {team: round(max(0, 26 - (i + 1)) * 2.4, 3) for i, team in enumerate(ap25)}

ncaaf_week1 = [
    ("Clemson", "LSU", "2026-09-05"),
    ("East Carolina", "Alabama", "2026-09-05"),
    ("Texas State", "Texas", "2026-09-05"),
    ("Boise State", "Oregon", "2026-09-05"),
]
ncaaf_games = [build_game("ncaaf", a, h, d, ncaaf_ratings.get(h, 0), ncaaf_ratings.get(a, 0)) for a, h, d in ncaaf_week1]
results["leagues"]["ncaaf"] = {
    "label": "NCAA Football",
    "status": "Week 1 — unranked opponent ratings treated as replacement level (0); scoped to AP Top 25 preseason poll",
    "ratings": ncaaf_ratings, "games": ncaaf_games,
}

# ---------------- NBA (offseason, ratings only) ----------------
nba_rows = read_csv("nba_2025_26_final.csv")
nba_ratings = {}
for r in nba_rows:
    w, l = int(r["w"]), int(r["l"])
    win_pct = w / (w + l)
    nba_ratings[r["team"]] = round((win_pct - 0.5) * 20, 3)

results["leagues"]["nba"] = {
    "label": "NBA",
    "status": "offseason — 2026-27 season tips off in October; ratings below are final 2025-26 power ratings (unregressed)",
    "ratings": nba_ratings, "games": [],
}

# ---------------- NCAA Basketball / Champions League (placeholders) ----------------
results["leagues"]["ncaab"] = {
    "label": "NCAA Basketball", "status": "offseason — season tips off in November 2026",
    "ratings": {}, "games": [],
}
results["leagues"]["ucl"] = {
    "label": "Champions League", "status": "league phase has not started — kicks off mid-September 2026",
    "ratings": {}, "games": [],
}

with open(os.path.join(DATA, "predictions.json"), "w") as f:
    json.dump(results, f, indent=2)

for lk, lv in results["leagues"].items():
    print(lk, lv["label"], "games:", len(lv["games"]), "teams rated:", len(lv["ratings"]))
    for g in lv["games"]:
        print("  ", g["away"], "@", g["home"], "->", g["home_win_prob"], "pick:", g["pick"], g["confidence"])
