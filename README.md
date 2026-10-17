# cariuang-bot: a Claude-built forecasting bot for the Metaculus FutureEval tournament

This repository is the open-source code of the Metaculus bot `cariuang-bot`
(bot-only tournaments: Fall 2026 FutureEval and MiniBench).

**Disclosure.** The bot was written by Claude (Anthropic's AI model) under the direction of its human owner
(an individual hobbyist in Indonesia), as part of an experiment in earning a first dollar with AI work.
The owner holds the Metaculus account and is responsible for it. Every forecast is made by the bot
itself through the Metaculus API; no human edits or pastes forecasts.

## What it does

1. Every 20 minutes a GitHub Actions job asks Metaculus for open questions in the Fall 2026 FutureEval
   tournament and the current MiniBench. Questions already forecast are skipped.
2. **Research:** Gemini with Google Search grounding writes a short brief (current status, upcoming dates,
   base rates, prediction-market or forecaster numbers for the same or a related event).
3. **Forecast:** several independent forecasts from different models (Gemini 2.5 Flash and DeepSeek),
   aggregated per question type (binary: mean of log-odds with a cap; numeric and date: pooled CDF;
   multiple choice: mean of options).
4. **Comment:** the bot leaves a private comment with its reasoning on every question it forecasts, as the rules require.

## Files

- `main.py`: the bot (a subclass of the `forecasting-tools` ForecastBot).
- `research.py`: web research.
- `.github/workflows/run_bot_on_tournament.yaml`: the schedule.

Derived from Metaculus' public `metac-bot-template`. Secrets (`METACULUS_TOKEN`, `GEMINI_API_KEY`,
`DEEPSEEK_API_KEY`) live in GitHub Actions secrets and are never committed.

Forecasts are probabilistic guesses, not advice.
