# Copyright (c) Microsoft. All rights reserved.

"""Conversation summarization agent for APO prompt optimization.

This module defines a summarization agent that generates call summaries using a tunable
prompt template. The prompt is evaluated by an LLM-as-judge grader (gpt-5.2) that scores
on a 0-100 scale. Data is loaded from S3 via Spark (Databricks environment).

Usage (debug, single rollout)::

    # In a Databricks notebook:
    import asyncio
    from conversation_summarization import debug_conversation_summarizer
    asyncio.run(debug_conversation_summarizer(dbutils, limit=1))

For APO training, see ``conversation_summarization_apo.py``.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Optional, TypedDict, cast

from openai import OpenAI
from pydantic import BaseModel, Field
from rich.console import Console

from agentlightning.adapter import TraceToMessages
from agentlightning.litagent import rollout
from agentlightning.reward import find_final_reward
from agentlightning.runner import LitAgentRunner
from agentlightning.store import InMemoryLightningStore
from agentlightning.tracer import OtelTracer
from agentlightning.types import Dataset, PromptTemplate

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Task types
# ---------------------------------------------------------------------------


class SummarizationTaskInput(TypedDict):
    call_conversation: str
    language: str
    additional_instructions: str


class SummarizationTask(TypedDict):
    id: str
    task_input: SummarizationTaskInput
    expected_summary: str  # ground truth (y_true from feedback/silver_feedback)


# ---------------------------------------------------------------------------
# Baseline prompt template (V8A)
# ---------------------------------------------------------------------------

# Copied from ai-copilot-prompt-benchmark summarization_prompt.py
# The {#if additional_instructions??}...{/if} conditional is removed — additional_instructions
# is always injected (empty string when absent from the data).
V8A_PROMPT = r"""You are an expert call center analyst. Your task is to create a comprehensive summary of a customer support call.

BEFORE WRITING THE SUMMARY, mentally identify:
1. The CLIENT's primary reason for calling
2. All distinct topics/issues discussed throughout the call (not just the conclusion)
3. Any emotions expressed by the CLIENT (frustration, confusion, satisfaction, urgency)
4. The AGENT's key responses, solutions offered, and actions taken
5. Any unresolved matters requiring follow-up

SUMMARY REQUIREMENTS:
- Cover the ENTIRE conversation chronologically, not just the resolution
- Give balanced attention to ALL topics discussed, regardless of when they appeared
- Include specific details: names, dates, account numbers, amounts, reference IDs mentioned
- Explicitly state the CLIENT's emotional state when evident (e.g., "The CLIENT expressed frustration about...")
- Document what the AGENT did to address each concern
- ONLY include information explicitly stated in the transcript - do not infer or fabricate details

NEXT ACTIONS:
- List concrete actions the AGENT must take after this call
- Each action should be specific and actionable
- If no follow-up is needed, return an empty list []

CRITICAL RULES:
- Write in {language}
- Never include information not present in the transcript
- If something is unclear in the transcript, do not guess - omit it
{additional_instructions}

OUTPUT FORMAT:
Respond ONLY with valid JSON, no markdown, no code blocks:
{{"summarization": "<comprehensive summary>", "next_actions": ["<action 1>", "<action 2>"]}}

TRANSCRIPT:
```
{call_conversation}
```
"""


def prompt_template_baseline() -> PromptTemplate:
    return PromptTemplate(
        template=V8A_PROMPT,
        engine="f-string",
    )


# ---------------------------------------------------------------------------
# LLM-as-a-judge grader
# ---------------------------------------------------------------------------

SUMMARIZATION_LLM_AS_A_JUDGE_PROMPT = r"""
# Task
You are provided with a **call transcript** and a **summarization** of that call.
Your goal is to **evaluate** how accurately and comprehensively the summarization reflects the content and meaning of the transcript, according to the following criteria:

1. **Full Conversation Coverage**
   - The summarization should summarize all parts of the conversation rather than focusing only on a single segment or the conclusion.

2. **Balanced Treatment of Topics**
   - The summarization should represent all topics discussed in the call fairly, without omitting or overemphasizing certain points.

3. **Essential Details**
   - The summarization should capture key issues, inquiries, and data points accurately, without oversimplification or omission.

4. **Agent Responses**
   - The summarization should adequately reflect the **AGENT**'s answers, guidance, or actions taken in response to the **CLIENT**'s concerns.

5. **Emotional Context**
   - The summarization should mention frustrations, doubts, or other emotions from the **CLIENT**, as well as how the **AGENT** addressed them.

6. **Accuracy / Hallucination**
   - The summarization should be factual and free from fabricated details or "hallucinations" not supported by the conversation.
   - **Note**: If any hallucination is detected, this should heavily lower the overall score, regardless of other strengths.

---

# Instructions

Follow these three steps:

## 1. **Analysis**
- Briefly review both the transcript and the summarization.
- Note how the summarization covers or misses each of the above criteria.
- Identify any significant omissions, distortions, or strong points related to its accuracy.

## 2. **Reasoning**
- Provide a concise explanation of how effectively the summarization meets the criteria.
- Highlight specific strengths or weaknesses.
- Mention whether there are any indications of hallucinated or fabricated content.

## 3. **Score**
- Assign a **score between 0 and 100** to reflect the summarization's overall alignment with the evaluation criteria:
  - **100** indicates a flawless summarization that addresses every criterion accurately and includes no hallucinatory content.
  - **0** indicates it completely fails to meet the criteria or is entirely fabricated.
  - If hallucinations are present, the score should be heavily reduced, regardless of other positive factors.

---

# call transcript
```
{conversation}
```

# summarization
```
{generated_summary}
```
"""


class JudgeResponse(BaseModel):
    reason: str = Field(description="Brief reasoning for the score.")
    score: int = Field(description="The score on a 0-100 scale.")


def summarization_grader(client: OpenAI, generated_summary: Optional[str], conversation: str) -> float:
    """Score a generated summary against the original conversation transcript.

    Uses an LLM-as-judge (o4-mini-data-curation) that evaluates coverage, balance, detail accuracy,
    agent response representation, emotional context, and hallucination.

    Returns a float in the 0-100 range.
    """
    judge_prompt = SUMMARIZATION_LLM_AS_A_JUDGE_PROMPT.format(
        conversation=conversation,
        generated_summary=generated_summary or "",
    )

    max_retries = 3
    for attempt in range(max_retries):
        judge = client.chat.completions.parse(
            model="o4-mini-data-curation",
            messages=[
                {"role": "user", "content": judge_prompt},
            ],
            response_format=JudgeResponse,
        )

        parsed = judge.choices[0].message.parsed
        if parsed is not None:
            return float(parsed.score)

        logger.warning("Judge returned unparseable response (attempt %d/%d)", attempt + 1, max_retries)

    raise ValueError(f"Judge failed to return valid response after {max_retries} attempts")


# ---------------------------------------------------------------------------
# Rollout function
# ---------------------------------------------------------------------------


@rollout
def conversation_summarizer(task: SummarizationTask, prompt_template: PromptTemplate) -> float:
    """Generate a call summary using the prompt template and grade it with the LLM judge.

    The prompt template is the artifact being optimized by APO. It is formatted with the
    task's ``call_conversation``, ``language``, and ``additional_instructions`` fields, then
    sent as a single-turn request to gpt-5-mini (reasoning_effort=minimal).
    """
    import os

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))

    # TODO: Remove debugging prints after APO template issues are resolved
    print(f"[DEBUG] Template engine: {prompt_template.engine}")
    print(f"[DEBUG] Full template:\n{prompt_template.template}")
    print(f"[DEBUG] Task input keys: {list(task['task_input'].keys())}")
    print(f"[DEBUG] language={task['task_input']['language']!r}")
    print(f"[DEBUG] additional_instructions={task['task_input']['additional_instructions']!r}")
    print(f"[DEBUG] call_conversation:\n{task['task_input']['call_conversation']}")

    try:
        user_message = prompt_template.format(**task["task_input"])
    except (KeyError, IndexError, ValueError) as e:
        # APO-rewritten templates may use single braces for JSON examples (e.g. {"key": "value"})
        # instead of escaped double braces ({{"key": "value"}}), causing str.format() to fail.
        print(f"[ERROR] prompt_template.format() failed: {type(e).__name__}: {e}")
        print(f"[ERROR] Full template:\n{prompt_template.template}")
        raise

    # TODO: Remove debugging print after APO template issues are resolved
    print(f"[DEBUG] Formatted message:\n{user_message}")

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": user_message},
        ],
        reasoning_effort="minimal",
    )

    generated_summary = resp.choices[0].message.content

    # TODO: Remove debugging print after APO template issues are resolved
    print(f"[DEBUG] Generated summary:\n{generated_summary}")

    return summarization_grader(client, generated_summary, task["task_input"]["call_conversation"])


# ---------------------------------------------------------------------------
# Data loading (S3 + Spark — Databricks environment)
# ---------------------------------------------------------------------------

# S3 constants (production EU bucket)
S3_BUCKET_PRD = "td-databricks-prd-eu-central-1-s3-aidatacuration"
S3_MOUNT_FOLDER = "/mnt/ai_data_curation/"
DATASETS_FOLDER = "datasets"


def _mount_s3(dbutils: object, s3_bucket: str, s3_mnt_folder: str) -> None:
    """Mount an S3 bucket in Databricks if not already mounted."""
    s3_source = f"s3a://{s3_bucket}"
    for m in dbutils.fs.mounts():  # type: ignore[union-attr]
        if Path(m.mountPoint).resolve() == Path(s3_mnt_folder).resolve():
            if m.source == s3_source:
                logger.debug("S3 bucket %s already mounted at %s", s3_bucket, s3_mnt_folder)
            else:
                dbutils.fs.unmount(s3_mnt_folder)  # type: ignore[union-attr]
                dbutils.fs.mount(s3_source, s3_mnt_folder)  # type: ignore[union-attr]
            return
    dbutils.fs.mount(s3_source, s3_mnt_folder)  # type: ignore[union-attr]


def load_summarization_tasks(
    dbutils: object,
    env: str = "prd",
    region: str = "eu",
    source_table: str = "observability",
    data_file: str = "20251210_000000-20251214_000000",
    num_samples: Optional[int] = 20,
    sample_seed: int = 42,
) -> Dataset[SummarizationTask]:
    """Load summarization tasks from S3 parquet via Spark.

    This is a self-contained data loader that replicates the loading logic from
    ``ai-copilot-prompt-benchmark`` without importing that package.

    Args:
        dbutils: Databricks ``dbutils`` object for S3 mounting.
        env: Environment name (``"prd"``, ``"stg"``, ``"qa"``).
        region: Data region (``"eu"``, ``"us"``).
        source_table: Source table in the data lake (``"defined_ai"`` or ``"observability"``).
        data_file: Root filename of the parquet + schema pair.
        num_samples: If set, randomly sample this many tasks.
        sample_seed: Random seed for reproducible sampling.

    Returns:
        A ``Dataset[SummarizationTask]`` list of tasks ready for APO.
    """
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import coalesce, col, expr, size
    from pyspark.sql.types import StructType

    # Mount S3
    s3_buckets = {
        "prd": S3_BUCKET_PRD,
        "stg": "td-databricks-stg-us-east-1-s3-aidatacuration",
        "qa": "td-databricks-qa-us-east-1-s3-aidatacuration",
    }
    s3_bucket = s3_buckets[env]
    _mount_s3(dbutils, s3_bucket, S3_MOUNT_FOLDER)

    # Build data paths
    # Path pattern: /mnt/ai_data_curation/datasets/{source_table}/{region}/summarization/
    data_dir = Path(f"/dbfs{S3_MOUNT_FOLDER}") / DATASETS_FOLDER / source_table / region / "summarization"
    spark_data_dir = Path(S3_MOUNT_FOLDER) / DATASETS_FOLDER / source_table / region / "summarization"

    schema_path = data_dir / f"{data_file}_schema.json"
    parquet_path = spark_data_dir / f"{data_file}.parquet"

    # Read schema and parquet
    spark = SparkSession.builder.getOrCreate()
    with open(str(schema_path), "r") as f:
        schema = StructType.fromJson(json.loads(f.read()))

    data = spark.read.schema(schema).parquet(str(parquet_path))

    # Filter: conversation_turns >= 7
    data = data.withColumn("conversation_turns", size(col("conversation_datapoint.data.messages"))).filter(
        col("conversation_turns") >= 7
    )

    # Add label columns (y_true from feedback/silver_feedback)
    columns = data.columns
    has_feedback = "feedback" in columns
    has_silver_feedback = "silver_feedback" in columns

    if has_feedback and has_silver_feedback:
        data = data.withColumn("y_true", coalesce(col("feedback")[0]["value"], col("silver_feedback")[0]["value"]))
    elif has_feedback:
        data = data.withColumn("y_true", col("feedback")[0]["value"])
    elif has_silver_feedback:
        data = data.withColumn("y_true", col("silver_feedback")[0]["value"])
    else:
        logger.warning("No feedback or silver_feedback columns found — y_true will be null")
        from pyspark.sql.functions import lit

        data = data.withColumn("y_true", lit(None))

    # Extract summarization-specific columns
    data = data.withColumn("conversation", expr("filter(io.inputs, x -> x.argument_id = 'call_conversation')[0].value"))
    data = data.withColumn("summary_language", expr("filter(io.inputs, x -> x.argument_id = 'language')[0].value"))
    data = data.withColumn(
        "additional_instructions",
        expr("filter(io.inputs, x -> x.argument_id = 'additional_instructions')[0].value"),
    )

    # Filter: exclude Arabic
    data = data.filter(col("summary_language") != "ar-SA")

    # Sample if requested
    if num_samples:
        count = data.count()
        fraction = min(1.0, num_samples / count)
        data = data.sample(withReplacement=False, fraction=fraction, seed=sample_seed).limit(num_samples)

    # Convert to pandas and build task list
    pdf = data.select("conversation", "summary_language", "additional_instructions", "y_true").toPandas()

    tasks: List[SummarizationTask] = []
    for idx, row in pdf.iterrows():
        tasks.append(
            SummarizationTask(
                id=str(idx),
                task_input=SummarizationTaskInput(
                    call_conversation=row["conversation"] or "",
                    language=row["summary_language"] or "",
                    additional_instructions=row["additional_instructions"] or "",
                ),
                expected_summary=row["y_true"] or "",
            )
        )

    logger.info("Loaded %d summarization tasks", len(tasks))
    return cast(Dataset[SummarizationTask], tasks)


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------


async def debug_conversation_summarizer(dbutils: object, limit: int = 1) -> None:
    """Run a small number of rollouts for debugging purposes."""
    runner = LitAgentRunner[SummarizationTask](OtelTracer())
    store = InMemoryLightningStore()
    prompt_template = prompt_template_baseline()
    tasks = load_summarization_tasks(dbutils, num_samples=limit)
    with runner.run_context(agent=conversation_summarizer, store=store):
        for task in tasks:
            console.print("[bold green]=== Task ===[/bold green]", task["id"], sep="\n")
            rollout_result = await runner.step(task, resources={"prompt_template": prompt_template})
            spans = await store.query_spans(rollout_result.rollout_id)
            adapter = TraceToMessages()
            messages = adapter.adapt(spans)
            for message_idx, message in enumerate(messages):
                console.print(f"[bold purple]=== Postmortem Message #{message_idx} ===[/bold purple]")
                console.print(json.dumps(message))
            reward = find_final_reward(spans)
            console.print("[bold purple]=== Postmortem Reward ===[/bold purple]", reward, sep="\n")


if __name__ == "__main__":
    # This script requires a Databricks environment with dbutils available.
    # For local debugging, mock dbutils or use a pre-loaded dataset.
    print("This module is designed for Databricks execution.")
    print("Use debug_conversation_summarizer(dbutils) in a Databricks notebook.")
