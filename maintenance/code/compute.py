import csv, math, json, os, sys, datetime
sys.path.insert(0, "/home/claude/sportspredict")
from engine import win_prob, shrink, make_pick, nfl_team_rating_v2, nfl_regime_change_sigma

DATA = "/home/claude/sportspredict/data"
TODAY = datetime.date.today().isoformat()

def read_csv(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))

with open(os.path.join(DATA, "model_params.json")) as f:
    PARAMS = json.load(f)

def P(league):
    return PARAMS["leagues"][league]

def build_game(league, away, home, date, rh, ra, note=None, sigma_override=None, model_version=None):
    p = P(league)
    sigma = sigma_override if sigma_override is not None else p["sigma"]
    home_prob = win_prob(rh, ra, p["home_adv"], sigma)
    g = {"away": away, "home": home, "date": date, "home_win_prob": round(home_prob, 3)}
    if note:
        g["note"] = note
    if model_version is not None:
        g["model_version"] = model_version
    g.update(make_pick(away, home, home_prob))
    return g

# NOTE on design: league blocks below fall into two shapes.
#   (a) "live" leagues (MLB/EPL/MLS/PLL fixtures) read their ratings/schedule
#       from data files (CSVs) that a refresh step regenerates from real
#       current sources each run -- this file itself never hardcodes a game
#       date or matchup for these.
#   (b) "point-in-time" leagues (NFL/NCAAF Week 1, PLL's regular-season
#       ratings, NBA/NCAAB/UCL off-season placeholders) are genuinely static
#       until their season progresses -- real, already-scheduled/final data,
#       not stand-ins for a live fetch -- so they stay as literals here on
#       purpose. Revisit NFL/NCAAF once Week 1 has been played.

results = {"generated_at": TODAY, "model_version": PARAMS["version"], "leagues": {}}

# ---------------- NFL (v2: win-total-baseline ratings + situational adjustments) ----------------
# v2 replaces v1's point-diff-carryover ratings, which backtested at Brier
# 0.263 on real 2024 games (worse than a coin flip) because they were never
# anchored to anything with real predictive content -- see "Team rating
# methodology v2" in sports-predictor-app.md for the full diagnosis. Baseline
# = each team's real 2026 sportsbook season win total run through
# nfl_win_total_baseline() (inside nfl_team_rating_v2); layered with injury/
# turnover adjustments where we have specific sourced data, and a sigma bump
# for any team in its first season under a new head coach. Every NFL game
# built here is tagged model_version 2 -- already-logged v1 picks (e.g. the
# locked Week 1 slate) are a separate, untouched record; this only affects
# what gets computed/logged from here on.

NFL_SEASON_WIN_TOTALS = {  # 2026 season win totals, sourced via WebSearch/WebFetch (FOX Sports/DraftKings-derived), 2026-09-07
    "Arizona Cardinals": 3.5, "Atlanta Falcons": 6.5, "Baltimore Ravens": 11.5,
    "Buffalo Bills": 10.5, "Carolina Panthers": 7.5, "Chicago Bears": 9.5,
    "Cincinnati Bengals": 10.5, "Cleveland Browns": 5.5, "Dallas Cowboys": 9.5,
    "Denver Broncos": 9.5, "Detroit Lions": 10.5, "Green Bay Packers": 9.5,
    "Houston Texans": 9.5, "Indianapolis Colts": 7.5, "Jacksonville Jaguars": 8.5,
    "Kansas City Chiefs": 10.5, "Las Vegas Raiders": 5.5, "Los Angeles Chargers": 9.5,
    "Los Angeles Rams": 11.5, "Miami Dolphins": 4.5, "Minnesota Vikings": 8.5,
    "New England Patriots": 10.5, "New Orleans Saints": 7.5, "New York Giants": 7.5,
    "New York Jets": 5.5, "Philadelphia Eagles": 10.5, "Pittsburgh Steelers": 8.5,
    "San Francisco 49ers": 9.5, "Seattle Seahawks": 10.5, "Tampa Bay Buccaneers": 8.5,
    "Tennessee Titans": 6.5, "Washington Commanders": 7.5,
}

NFL_NEW_HC_TEAMS = {  # first season under a new head coach, 2026 -- drives the regime-change sigma bump
    "Buffalo Bills", "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens",
    "Cleveland Browns", "Las Vegas Raiders", "Miami Dolphins", "New York Giants",
    "Pittsburgh Steelers", "Tennessee Titans",
}

# Injury/turnover adjustments where we have specific, sourced data (currently
# just the Ravens -- see ravens-data.json). Teams not listed here get 0 for
# both, an honest default: the win-total baseline already does nearly all the
# work per the Ravens diagnostic. Extend opportunistically as real injury/
# transaction news gets sourced for other teams.
NFL_INJURY_TURNOVER = {
    "Baltimore Ravens": {
        "injuries": [
            {"pos": "C", "severity": "out"},              # Danny Pinter, out extended time
            {"pos": "DT", "severity": "positive_return"},  # Nnamdi Madubuike, back to full-contact practice
        ],
        "departures": ["TE", "FB", "OT", "P", "S", "CB", "OLB", "WR"],  # Likely, Ricard, Faalele, Stout, Washington, Alexander, Oweh, C.Johnson
        "acquisitions": ["OLB", "S", "CB", "LB", "ILB"],                 # D.Jones, Gilman, White, Hendrickson, Barrett
    },
}

nfl_ratings = {}
for team, win_total in NFL_SEASON_WIN_TOTALS.items():
    extra = NFL_INJURY_TURNOVER.get(team, {})
    nfl_ratings[team] = round(nfl_team_rating_v2(
        win_total,
        injuries=extra.get("injuries"),
        departures=extra.get("departures"),
        acquisitions=extra.get("acquisitions"),
    ), 3)

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
nfl_games = [
    build_game(
        "nfl", a, h, d, nfl_ratings.get(h, 0), nfl_ratings.get(a, 0),
        sigma_override=nfl_regime_change_sigma(
            P("nfl")["sigma"],
            home_new_coach=h in NFL_NEW_HC_TEAMS,
            away_new_coach=a in NFL_NEW_HC_TEAMS,
        ),
        model_version=2,
    )
    for a, h, d in nfl_week1
]
results["leagues"]["nfl"] = {
    "label": "NFL",
    "status": "v2 ratings (win-total baseline + injury/turnover/regime-change adjustments) — Week 1 not yet played",
    "model_version": 2,
    "ratings": nfl_ratings, "games": nfl_games,
}

# ---------------- MLB (live: fresh standings + slate each refresh) ----------------
mlb_rows = read_csv("mlb_2026_standings.csv")
mlb_ratings = {}
for r in mlb_rows:
    w, l = int(r["w"]), int(r["l"])
    gp = w + l
    raw = int(r["run_diff"]) / gp
    mlb_ratings[r["team"]] = round(shrink(raw, gp, P("mlb")["shrink_k"]), 3)

mlb_slate = [(r["away"], r["home"], r["date"]) for r in read_csv("mlb_slate.csv")]
mlb_games = [build_game("mlb", a, h, d, mlb_ratings.get(h, 0), mlb_ratings.get(a, 0)) for a, h, d in mlb_slate]
results["leagues"]["mlb"] = {"label": "MLB", "status": "in season", "ratings": mlb_ratings, "games": mlb_games}

# ---------------- Premier League (live) ----------------
epl_rows = read_csv("epl_2026_27_standings.csv")
epl_ratings = {}
for r in epl_rows:
    gp = int(r["gp"])
    raw = int(r["gd"]) / gp if gp else 0
    epl_ratings[r["team"]] = round(shrink(raw, gp, P("epl")["shrink_k"]), 3)

epl_fixtures = [(r["away"], r["home"], r["date"]) for r in read_csv("epl_fixtures.csv")]
epl_games = [build_game("epl", a, h, d, epl_ratings.get(h, 0), epl_ratings.get(a, 0)) for a, h, d in epl_fixtures]
results["leagues"]["epl"] = {
    "label": "Premier League",
    "status": "in season — early-season ratings are heavily shrunk toward league average",
    "ratings": epl_ratings, "games": epl_games,
}

# ---------------- MLS (live) ----------------
mls_rows = read_csv("mls_2026_standings.csv")
mls_ratings = {}
for r in mls_rows:
    gp = int(r["gp"])
    raw = int(r["gd"]) / gp if gp else 0
    mls_ratings[r["team"]] = round(shrink(raw, gp, P("mls")["shrink_k"]), 3)

mls_fixtures = [(r["away"], r["home"], r["date"]) for r in read_csv("mls_fixtures.csv")]
mls_games = [build_game("mls", a, h, d, mls_ratings.get(h, 0), mls_ratings.get(a, 0)) for a, h, d in mls_fixtures]
results["leagues"]["mls"] = {"label": "MLS", "status": "in season", "ratings": mls_ratings, "games": mls_games}

# ---------------- PLL Lacrosse (regular-season ratings are final/static; fixtures are live) ----------------
pll_ratings = {
    "Philadelphia Waterdogs": 2.5, "Boston Cannons": 0.833, "Maryland Whipsnakes": -0.833,
    "New York Atlas": -2.5, "Utah Archers": 0.833, "California Redwoods": 0.0,
    "Denver Outlaws": 0.0, "Carolina Chaos": -0.833,
}
pll_fixtures = [(r["away"], r["home"], r["date"], r["note"]) for r in read_csv("pll_fixtures.csv")]
pll_games = [build_game("pll", a, h, d, pll_ratings.get(h, 0), pll_ratings.get(a, 0), note) for a, h, d, note in pll_fixtures]
results["leagues"]["pll"] = {
    "label": "PLL Lacrosse",
    "status": "playoffs (semifinals) — neutral site, no home advantage applied",
    "ratings": pll_ratings, "games": pll_games,
}

# ---------------- NCAA Football (point-in-time: real Week 1 schedule, AP Top 25 preseason poll) ----------------
ap25 = ["Ohio State","Oregon","Georgia","Notre Dame","Texas","Indiana","Miami","Texas A&M",
        "Ole Miss","Oklahoma","LSU","Texas Tech","Alabama","BYU","USC","Michigan","Washington",
        "Penn State","SMU","Tennessee","Utah","Iowa","Houston","Louisville","Missouri"]
ncaaf_ratings = {team: round(max(0, 26 - (i + 1)) * 2.4, 3) for i, team in enumerate(ap25)}

# Week 1 (2026-09-05) has been played and graded (all 4 favorites covered — see
# predictions-history.json). The Week-2+ file-driven pipeline (CSV-based
# standings/fixtures like MLB/EPL/MLS) hasn't been built yet -- flagged for
# Zak rather than guessing at a new hardcoded slate with no real ranking
# update methodology behind it. Leaving games empty (rather than re-showing
# the finished Week 1 slate as if it were still upcoming) until that's built.
ncaaf_games = []
results["leagues"]["ncaaf"] = {
    "label": "NCAA Football",
    "status": "Week 1 complete (all graded) — Week 2+ needs a file-driven pipeline (not yet built); scoped to AP Top 25 preseason poll",
    "ratings": ncaaf_ratings, "games": ncaaf_games,
}

# ---------------- NBA (offseason, ratings only, static until 2026-27 tips off) ----------------
nba_ratings = {"Oklahoma City Thunder": 5.61, "San Antonio Spurs": 5.122, "Detroit Pistons": 4.634, "Denver Nuggets": 3.171, "Los Angeles Lakers": 2.927, "Boston Celtics": 3.659, "New York Knicks": 2.927, "Houston Rockets": 2.683, "Cleveland Cavaliers": 2.683, "Minnesota Timberwolves": 1.951, "Toronto Raptors": 1.22, "Atlanta Hawks": 1.22, "Portland Trail Blazers": 0.244, "Los Angeles Clippers": 0.244, "Philadelphia 76ers": 0.976, "Orlando Magic": 0.976, "Phoenix Suns": 0.976, "Charlotte Hornets": 0.732, "Miami Heat": 0.488, "Golden State Warriors": -0.976, "Milwaukee Bucks": -2.195, "Chicago Bulls": -2.439, "New Orleans Pelicans": -3.659, "Dallas Mavericks": -3.659, "Memphis Grizzlies": -3.902, "Sacramento Kings": -4.634, "Utah Jazz": -4.634, "Indiana Pacers": -5.366, "Washington Wizards": -5.854, "Brooklyn Nets": -5.122}

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
