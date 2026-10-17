"""cariuang-bot: Metaculus FutureEval bot (Fall 2026 tournament and MiniBench).

Written by Claude (Anthropic) for its human owner. Derived from Metaculus' public metac-bot-template
(kept unchanged in template_main.py). Differences from the template:

- Research: Gemini with Google Search grounding (research.py) instead of AskNews/OpenRouter.
- Forecasts: a small ensemble that mixes Gemini 2.5 Flash and DeepSeek, aggregated by the median.
- Binary answers are read with a regex first (saves one parser call per forecast).
- The prompts tell the forecaster to anchor on market/forecaster estimates found in the research.
- Comments are compact (the Metaculus rules treat long, low-value bot comments as spam).
"""

import argparse
import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import Literal

import dotenv

dotenv.load_dotenv()
# The local .env spells the DeepSeek variable differently from the name litellm expects.
if not os.getenv("DEEPSEEK_API_KEY") and os.getenv("DEEPSEEK_APIKEY"):
    os.environ["DEEPSEEK_API_KEY"] = os.environ["DEEPSEEK_APIKEY"]

from bot_helpers import (  # noqa: E402
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (  # noqa: E402
    BinaryPrediction,
    BinaryQuestion,
    DatePercentile,
    DateQuestion,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

from research import build_brief  # noqa: E402
from template_main import FallTemplateBot2026  # noqa: E402

logger = logging.getLogger(__name__)

P_MIN, P_MAX = 0.02, 0.98
_PCTS = (10, 20, 40, 60, 80, 90)
_NUM_RE = r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|[-+]?\.\d+"
_UNIT_WORDS = re.compile(r"million|billion|trillion|thousand|percent|%|k\b|bn\b", re.IGNORECASE)
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})(?:T(\d{2}:\d{2}(?::\d{2})?)Z?)?")
_PROB_RE = re.compile(r"Probability:\s*\**\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE)

ANCHOR_NOTE = (
    "If the research reports a prediction-market or forecaster estimate for this exact event, start from it and move away "
    "only for specific, stated reasons. If it reports none, start from the base rate of similar events."
)


class ThrottledLlm(GeneralLlm):
    """GeneralLlm with a shared concurrency cap so the free Gemini tier is not flooded."""

    _sem: asyncio.Semaphore | None = None

    async def invoke(self, prompt):  # type: ignore[override]
        if ThrottledLlm._sem is None:
            ThrottledLlm._sem = asyncio.Semaphore(2)
        async with ThrottledLlm._sem:
            return await super().invoke(prompt)


class ChainLlm:
    """Tries several models in order; used so a spent free-tier quota falls back to the next model."""

    def __init__(self, models):
        self.models = models

    async def invoke(self, prompt):
        last: Exception | None = None
        for model in self.models:
            try:
                return await model.invoke(prompt)
            except Exception as e:  # noqa: BLE001
                last = e
                logger.warning("model %s failed (%s); trying the next one", getattr(model, "model", "?"), type(e).__name__)
        assert last is not None
        raise last


class CariuangBot(FallTemplateBot2026):
    """Template bot with different research, a mixed-model ensemble and compact comments."""

    _structure_output_validation_samples = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ds = lambda t: GeneralLlm(model="deepseek/deepseek-chat", temperature=t, timeout=240, allowed_tries=2)  # noqa: E731
        gm = lambda name, t: ThrottledLlm(model=f"gemini/{name}", temperature=t, timeout=240, allowed_tries=1)  # noqa: E731
        self._pool = [
            ChainLlm([ds(0.3), gm("gemini-3.8-flash", 0.3)]),
            ChainLlm([gm("gemini-3.8-flash", 0.5), ds(0.5)]),
            ChainLlm([ds(0.6), gm("gemini-3.5-flash", 0.5)]),
            ChainLlm([gm("gemini-3.5-flash", 0.3), ds(0.4)]),
        ]
        self._rr = 0

    # Rotate the "default" forecaster over the pool: each question's predictions cover every entry once.
    def get_llm(self, purpose: str = "default", guarantee_type=None):  # type: ignore[override]
        if purpose == "default" and guarantee_type == "llm":
            llm = self._pool[self._rr % len(self._pool)]
            self._rr += 1
            return llm
        return super().get_llm(purpose, guarantee_type)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = await build_brief(
                question.question_text,
                question.background_info or "",
                question.resolution_criteria or "",
                question.fine_print or "",
            )
            logger.info("Research for %s (%d chars)", question.page_url, len(research))
            return research

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str):
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {ANCHOR_NOTE}
            {self._get_conditional_disclaimer_if_necessary(question)}

            Keep the whole answer under 350 words.
            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(self, question: BinaryQuestion, prompt: str):
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info("Reasoning for %s: %s", question.page_url, reasoning)
        matches = _PROB_RE.findall(reasoning)
        if matches:
            probability = float(matches[-1]) / 100.0
        else:
            parsing_instructions = clean_indents(
                f"""
                The text given to you is trying to give a probability forecast for a binary question.
                {self._create_resolved_question_parsing_message()}
                """
            )
            parsed: BinaryPrediction = await structure_output(
                reasoning,
                BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
                additional_instructions=parsing_instructions,
            )
            probability = parsed.prediction_in_decimal
        probability = max(P_MIN, min(P_MAX, probability))
        logger.info("Forecasted %s with prediction %.3f", question.page_url, probability)
        return ReasonedPrediction(prediction_value=probability, reasoning=reasoning)

    # ------------------------------------------------------------------ regex-first parsing (falls back to the LLM parser)

    @staticmethod
    def _last_percentile_lines(text: str) -> dict[int, str]:
        found: dict[int, str] = {}
        for m in re.finditer(r"Percentile\s*(\d{1,2})\s*:\s*(.+)", text):
            if int(m.group(1)) in _PCTS:
                found[int(m.group(1))] = m.group(2).strip()
        return found

    def _regex_numeric(self, text: str) -> list[Percentile] | None:
        lines = self._last_percentile_lines(text)
        if set(lines) != set(_PCTS):
            return None
        out = []
        for pct in _PCTS:
            line = lines[pct]
            if _UNIT_WORDS.search(line):
                return None  # scaled units: let the LLM parser interpret them
            m = re.match(_NUM_RE, line.replace("$", "").replace("~", "").strip())
            if not m:
                return None
            out.append(Percentile(percentile=pct / 100, value=float(m.group(0).replace(",", ""))))
        values = [p.value for p in out]
        return out if values == sorted(values) else None

    def _regex_date(self, text: str) -> list[Percentile] | None:
        lines = self._last_percentile_lines(text)
        if set(lines) != set(_PCTS):
            return None
        out = []
        for pct in _PCTS:
            m = _DATE_RE.search(lines[pct])
            if not m:
                return None
            stamp = f"{m.group(1)}T{m.group(2) or '00:00'}"
            if stamp.count(":") == 1:
                stamp += ":00"
            when = datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)
            out.append(Percentile(percentile=pct / 100, value=when.timestamp()))
        values = [p.value for p in out]
        return out if values == sorted(values) else None

    @staticmethod
    def _regex_options(text: str, options: list[str]) -> PredictedOptionList | None:
        probs = []
        for opt in options:
            hits = re.findall(
                r"(?:^|\n)\W*(?:Option\s*)?" + re.escape(opt) + r"\W*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", text, re.IGNORECASE
            )
            if not hits:
                return None
            probs.append(float(hits[-1]) / 100)
        total = sum(probs)
        if not 0.9 <= total <= 1.1:
            return None
        probs = [max(p, 0.01) for p in probs]
        total = sum(probs)
        probs = [p / total for p in probs]
        probs[-1] += 1.0 - sum(probs)
        return PredictedOptionList(
            predicted_options=[PredictedOption(option_name=o, probability=p) for o, p in zip(options, probs)]
        )

    async def _multiple_choice_prompt_to_forecast(self, question: MultipleChoiceQuestion, prompt: str):
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info("Reasoning for %s: %s", question.page_url, reasoning)
        prediction = self._regex_options(reasoning, list(question.options))
        if prediction is None:
            parsing_instructions = clean_indents(
                f"""
                Make sure that all option names are one of the following:
                {question.options}

                The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
                Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
                {self._create_resolved_question_parsing_message()}
                """
            )
            prediction = await structure_output(
                text_to_structure=reasoning,
                output_type=PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
                additional_instructions=parsing_instructions,
            )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _numeric_prompt_to_forecast(self, question: NumericQuestion, prompt: str):
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info("Reasoning for %s: %s", question.page_url, reasoning)
        percentile_list = self._regex_numeric(reasoning)
        if percentile_list is None:
            percentile_list = await structure_output(
                reasoning,
                list[Percentile],
                model=self.get_llm("parser", "llm"),
                additional_instructions=clean_indents(
                    f"""
                    The text given to you is trying to give a forecast distribution for a numeric question.
                    - This text is trying to answer the numeric question: "{question.question_text}".
                    {self._create_single_distribution_parsing_message(question)}
                    {self._create_resolved_question_parsing_message()}
                    - The units for the forecast are: {question.unit_of_measure}
                    - Give the values in those units (for example $500,000,000 with units "B $" is 0.5). Turn scientific notation into regular numbers.
                    - If percentiles are not explicitly given, do not return a parsed output.
                    """
                ),
                num_validation_samples=self._structure_output_validation_samples,
            )
        prediction = NumericDistribution.from_question(percentile_list, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _date_prompt_to_forecast(self, question: DateQuestion, prompt: str):
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info("Reasoning for %s: %s", question.page_url, reasoning)
        percentile_list = self._regex_date(reasoning)
        if percentile_list is None:
            date_percentiles = await structure_output(
                reasoning,
                list[DatePercentile],
                model=self.get_llm("parser", "llm"),
                additional_instructions=clean_indents(
                    f"""
                    The text given to you is trying to give a forecast distribution for a date question.
                    - This text is trying to answer the question: "{question.question_text}".
                    {self._create_single_distribution_parsing_message(question)}
                    {self._create_resolved_question_parsing_message()}
                    - Format the output as valid datetimes; assume midnight UTC if no hour is given.
                    - If percentiles are not explicitly given, do not return a parsed output.
                    """
                ),
                num_validation_samples=self._structure_output_validation_samples,
            )
            percentile_list = [Percentile(percentile=d.percentile, value=d.value.timestamp()) for d in date_percentiles]
        prediction = NumericDistribution.from_question(percentile_list, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_comment(self, question, research_prediction_collections, aggregated_prediction, final_cost, time_spent_in_minutes):  # type: ignore[override]
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        report_type = DataOrganizer.get_report_type_for_question_type(type(question))
        final = report_type.make_readable_prediction(aggregated_prediction)
        summary = (
            "# SUMMARY\n"
            "cariuang-bot (built by Claude; every forecast is made by the bot through the API)\n"
            f"*Question*: {question.question_text}\n"
            f"*Final Prediction*: {final}"
        )
        research_parts, forecast_parts = [], []
        for collection in research_prediction_collections:
            research_parts.append((collection.research_report or "").strip()[:1800])
            for j, pred in enumerate(collection.predictions, start=1):
                text = (pred.reasoning or "").strip()
                forecast_parts.append(
                    f"## Forecaster {j}: {report_type.make_readable_prediction(pred.prediction_value)}\n{text[-900:].split(' ', 1)[-1]}"
                )
        parts = [summary, "# RESEARCH\n" + "\n\n".join(research_parts), "# FORECASTS\n" + "\n\n".join(forecast_parts)]
        return "\n\n".join(parts)[:12000]


def build_bot(publish: bool) -> CariuangBot:
    return CariuangBot(
        research_reports_per_question=1,
        predictions_per_research_report=4,
        use_research_summary_to_forecast=False,
        enable_summarize_research=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=False,
        llms={
            "default": GeneralLlm(model="deepseek/deepseek-chat", temperature=0.3, timeout=240, allowed_tries=2),
            "summarizer": "deepseek/deepseek-chat",
            "researcher": "None",
            "parser": GeneralLlm(model="deepseek/deepseek-chat", temperature=0.0, timeout=120, allowed_tries=2),
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Run cariuang-bot")
    parser.add_argument("--mode", choices=["tournament", "test_questions"], default="tournament")
    parser.add_argument("--no-publish", action="store_true", help="forecast but do not post to Metaculus")
    args = parser.parse_args()
    run_mode: Literal["tournament", "test_questions"] = args.mode
    publish = not args.no_publish

    check_environment(strict=True)
    print_startup_banner(run_mode, will_publish=publish)
    bot = build_bot(publish)
    client = MetaculusClient()
    tournament_urls = {
        "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }
    if run_mode == "tournament":
        reports = asyncio.run(bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True))
        reports += asyncio.run(bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True))
    else:
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(bot.forecast_on_tournament("bot-testing-area", return_exceptions=True))
    bot.log_report_summary(reports)
    print_run_summary_banner(reports, will_publish=publish, tournament_url=tournament_urls[run_mode])
