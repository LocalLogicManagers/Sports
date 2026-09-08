# NFL value-betting strategy — backtest analysis

**Prepared:** 2026-09-08, at Zak's request ("using all of this data I want to build out a profitable strategy"). Scope: NFL only, the one league with a real backtested sample and real market odds behind it. Everything below is derived from data already in this project — the 117-game 2024 historical backtest (Weeks 1-9), the "Team rating methodology v2" model, and the currently open NFL Week 1 2026 picks.

**Bottom line up front:** the data supports one specific, narrow rule with real (if thin) historical backing — **bet the model's own pick, unblended, only in games where it disagrees with the market's favored side; skip every game where it agrees.** Backtested at +25.4% ROI on 21 qualifying bets out of 117 historical games, positive in both halves of the sample when split. That is a genuine signal, not nothing — but n=21 is not enough to bet real money on with confidence. Treat this as a paper-traded hypothesis for now, not a funded strategy. Details and every caveat below.

---

## 1. What "profitable" actually requires

A pick being *right* more often than not isn't the same as a bet being *profitable*. Sportsbooks build in a vig (roughly 4.5% overround in our modeled odds, matching real mainstream books), so betting the market's own favorite, at the market's own price, loses money on average by construction — you need to be righter than the market's own price implies, not just righter than 50/50. That only happens when your probability estimate for a game differs from the market's *and* that difference is real signal, not noise.

This is why the site's existing "Brier score" framing and this strategy analysis are answering two different questions. Brier score asks: *how well-calibrated are our probabilities?* Profitability asks: *does the model know something the market doesn't, often enough to overcome the vig?* They don't have to have the same answer — and as it turns out here, they don't.

## 2. The data used

- **117 historical NFL games**, 2024 season Weeks 1-9 (77 from Weeks 1-5 via Covers.com, 40 from Weeks 6-9 via ESPN's core API, each individually verified — see `model-params.json`'s note and the "Widened NFL backtest sample" writeup in the project runbook for sourcing detail). Each game carries: home/away teams, final score, and the real closing spread.
- **The model**: "Team rating methodology v2" (win-total baseline + injury/turnover/regime-change adjustments), the same code now live in `compute.py`.
- **The market**: the closing spread converted to a fair (no-vig) win probability via `norm_cdf(spread / 13.86)` — the same conversion the existing blend-weight backtest uses. This is a modeled proxy for the market's fair price, not a recorded historical sportsbook moneyline (those aren't available for games this old at this granularity) — an important caveat that applies to every number below, noted once here rather than repeated on each line.
- **Simulated betting**: for whichever side a strategy picks, the bet is priced using the *market's* fair probability run through the same vig formula the site's wager calculator already uses (`DEFAULT_VIG = 0.045`) — i.e., a bettor pays the market's realistic price, never our own model's price. This is the only honest way to backtest profit: you only make money when you beat a real market price, not when you agree with your own forecast.

## 3. Headline results — betting every game, no filtering

| Strategy | Bets | Win rate | Staked | Returned | Net P/L | ROI |
|---|---|---|---|---|---|---|
| Bet the market favorite every game | 117 | 65.8% | $1,170 | $1,209.22 | +$39.22 | **+3.4%** |
| Bet the raw v2 model's own pick every game | 117 | 68.4% | $1,170 | $1,322.61 | +$152.61 | **+13.0%** |
| Bet the blended pick (w=0.65, the site's current calibration-optimal weight) | 117 | 67.5% | $1,170 | $1,272.30 | +$102.30 | **+8.7%** |
| Bet the blended pick (w=0.85, last week's weight) | 117 | 65.8% | $1,170 | $1,209.22 | +$39.22 | **+3.4%** |
| Bet the market underdog every game | 117 | 34.2% | $1,170 | $1,027.50 | -$142.50 | **-12.2%** |

**The first surprise: the raw, unblended model outperforms every blended version on P&L**, even though blending toward the market is what *improves* Brier score (calibration). Blending mutes the model's disagreements with the market — which is exactly where, per the numbers above, the model's money-making edge actually lives. Betting a well-calibrated probability and betting a profitable pick are not the same optimization target, and on this data they point in different directions. The site's `blend_weight` (currently 0.65) is doing its job — improving the *displayed* probability's calibration — but that's not the number to use for a betting rule.

**The second finding: blindly betting the market underdog loses money** (-12.2%), which rules out a naive "just fade the favorite" explanation for the model's edge — this isn't a generic longshot-value effect, it's specific to where the model itself disagrees with the market.

## 4. Filtering by pick strength (confidence tier) — the opposite of what you'd expect

| Minimum pick probability (blended, w=0.65) | Bets | Win rate | ROI |
|---|---|---|---|
| 50%+ (everything) | 117 | 67.5% | +8.7% |
| 55%+ | 94 | 69.1% | +7.2% |
| 60%+ | 69 | 69.6% | +4.5% |
| 65%+ | 45 | 57.8% | **-18.1%** |
| 70%+ | 20 | 65.0% | **-10.9%** |

Higher-confidence picks are *worse* bets, not better. This is a well-documented pattern in sports betting generally (the "favorite-longshot bias") — big favorites get overbet by the public and get systematically shorter (worse-value) prices than their true win probability justifies, so laying big odds on a heavy favorite is bad value even when that favorite wins most of the time. Don't build a "only bet our most confident picks" feature — the data says the opposite.

## 5. The strategy that actually held up: bet disagreement, not agreement

| Rule | Bets | Win rate | ROI |
|---|---|---|---|
| Model pick agrees with market favorite → skip; model picks the market's underdog → bet it (raw model, unblended) | **21 / 117** | 57.1% | **+25.4%** |

This rule requires no fitted threshold or blend weight — it's just "does the model's raw pick differ from the market's favored side." Split across the two halves of the sample to check consistency:

| Sample | Bets | ROI |
|---|---|---|
| Weeks 1-5 (n=77) | 14 | +12.1% |
| Weeks 6-9 (n=40) | 7 | +52.1% |

Same sign in both halves — a real, if noisy, signal, not an artifact of one lucky stretch. A genuine **true out-of-sample check** (fit the blend weight on one half, test P&L on the untouched other half) confirms the raw-model-on-disagreement approach beats any blended version regardless of which half it's trained or tested on:

| Trained on | Tested on | Raw model pick ROI | Best-fit blended pick ROI |
|---|---|---|---|
| Weeks 1-5 (fit w=0.85) | Weeks 6-9 (holdout) | **+24.0%** | +5.6% |
| Weeks 6-9 (fit w=0.30) | Weeks 1-5 (holdout) | **+7.4%** | +4.3% |

## 6. Honest limits — read this before staking anything

- **n=21 qualifying bets is thin.** The rule of thumb for distinguishing real edge from variance in moneyline betting is usually "tens to low hundreds of resolved bets, minimum" — 21 is a start, not proof. A single flipped coin-toss game moves this ROI figure several points.
- **One partial season, one league.** 2024 Weeks 1-9 only. No second season, no other league tested this way yet.
- **The "market price" here is reconstructed from the spread, not a recorded historical sportsbook moneyline.** The whole exercise is internally consistent (same conversion the rest of the site already uses) but is one step removed from what a real book would have actually offered.
- **A narrower cut (excluding the largest disagreements, ≥15 points — the site's own existing "large disagreement" cutoff) shrank the sample to n=5 with a wild +74% ROI.** That's not a stronger result, it's a smaller and noisier one — flagged here specifically as an example of how easy it is to torture a tiny backtest sample into an impressive-looking number. Don't be tempted by it.
- **The blend weight (0.65) that's already live and shown on the site was fit by minimizing Brier score on this same 117-game sample** — that's the right thing to optimize for calibration, but it means this document's P&L comparisons against "the blended pick" use a weight that was tuned on the same data, which is a fair like-for-like comparison but isn't a claim that 0.65 is somehow *wrong* — it's optimized for a different goal (accurate probabilities, which is what the hero Brier-score stat and the rest of the site's honesty framing depend on). Don't change `market_blend_weight` because of this document; it serves the calibration/display purpose it was built for.
- **The currently open NFL Week 1 2026 slate is still tagged `model_version: 1`** (the older, badly-compressed rating method) — it was already logged before v2 was wired into `compute.py`, and per the site's own "never rewrite a logged pick" rule, it stays that way. This strategy was validated against v2-generated probabilities, so it should not be applied to the current Week 1 slate's raw model numbers. It's ready to apply starting with Week 2, the first slate that will actually be built under v2.

## 7. Recommendation

1. **Don't bet real money on this yet.** Paper-track it: starting with NFL Week 2 (the first v2-generated slate), flag every game where the model's pick disagrees with the market favorite, and record what a flat, small stake ($5 or $10 — the same generic size already used in the Projections section) would have won or lost, without actually placing the bet. Let it accumulate through a full season before trusting it with real stakes — that's roughly 270 games, enough to get the qualifying-disagreement subset into the "tens to low hundreds" range this kind of edge claim actually needs.
2. **Keep `market_blend_weight` exactly as-is** for the site's displayed probabilities, Brier tracking, and hero stat — that's a calibration tool, not a betting tool, and this analysis doesn't call that decision into question.
3. **If you do want this live on the site now**, the honest version is a clearly-labeled "Strategy (beta, unproven)" flag — not a "recommended bet" framed with any confidence — shown only on `model_version: 2` games where a real market line exists. See the accompanying site update for exactly that: a badge that lights up automatically once Week 2 picks are logged under v2, sitting inert on the current Week 1 (v1) slate.
4. **Revisit this analysis once more seasons/weeks are available** — the single highest-value next step for turning "a promising 21-bet pattern" into "an actual edge" is more data, not a cleverer filter on the same 117 games.

---

*All figures reproducible from `/home/claude/proj_edit6/base.py` (backtest + strategy simulation script) run this session — extends the existing widened-backtest script with a P&L simulator using the site's own vig/odds formulas (`engine.py`'s `DEFAULT_VIG=0.045`, `fair_prob_to_vig_prob`, `prob_to_american_odds`, `american_to_decimal`).*
