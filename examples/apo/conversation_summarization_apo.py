# Copyright (c) Microsoft. All rights reserved.

"""APO training script for conversation summarization prompt optimization.

This script uses Agent Lightning's Automatic Prompt Optimization (APO) algorithm to
iteratively improve the V8A summarization prompt. It loads production data from S3 via
Spark, splits into train/val, and runs beam search optimization.

Designed for execution in a Databricks notebook::

    # In a Databricks notebook cell:
    %run ./conversation_summarization_apo

    # Or import and call:
    from conversation_summarization_apo import main
    main(dbutils)
"""

import logging
from typing import Tuple, cast

from openai import AsyncOpenAI
from room_selector_apo import print_training_results, setup_apo_logger

from agentlightning import Trainer, setup_logging
from agentlightning.adapter import TraceToMessages
from agentlightning.algorithm.apo import APO
from agentlightning.types import Dataset

from conversation_summarization import (
    SummarizationTask,
    conversation_summarizer,
    load_summarization_tasks,
    prompt_template_baseline,
)

logger = logging.getLogger(__name__)


def load_train_val_dataset(
    dbutils: object,
    num_samples: int | None = None,
) -> Tuple[Dataset[SummarizationTask], Dataset[SummarizationTask]]:
    """Load summarization tasks and split 50/50 into train and validation sets."""
    dataset_full = load_summarization_tasks(dbutils, num_samples=num_samples)
    train_split = len(dataset_full) // 2
    dataset_train = [dataset_full[i] for i in range(train_split)]
    dataset_val = [dataset_full[i] for i in range(train_split, len(dataset_full))]
    return cast(Dataset[SummarizationTask], dataset_train), cast(Dataset[SummarizationTask], dataset_val)


def main(dbutils: object) -> None:
    """Run APO optimization on the conversation summarization prompt.

    Args:
        dbutils: Databricks ``dbutils`` object for S3 data access.
    """
    import os

    setup_logging()
    setup_apo_logger()

    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))

    algo = APO[SummarizationTask](
        openai_client,
        gradient_model="gpt-5.2",
        apply_edit_model="gpt-5-mini",
        val_batch_size=20,
        gradient_batch_size=4,
        beam_width=4,
        branch_factor=4,
        beam_rounds=3,
        _poml_trace=True,
    )

    trainer = Trainer(
        algorithm=algo,
        n_runners=4,
        initial_resources={
            "prompt_template": prompt_template_baseline(),
        },
        adapter=TraceToMessages(),
    )

    dataset_train, dataset_val = load_train_val_dataset(dbutils)
    logger.info("Train: %d tasks, Val: %d tasks", len(dataset_train), len(dataset_val))

    trainer.fit(agent=conversation_summarizer, train_dataset=dataset_train, val_dataset=dataset_val)

    print_training_results(algo)

    print("\n" + "=" * 90)
    print("  BEST PROMPT TEMPLATE")
    print("=" * 90)
    print(algo.get_best_prompt().template)
    print("=" * 90)


if __name__ == "__main__":
    # Assumes Databricks environment with dbutils in global scope
    main(dbutils)  # type: ignore[name-defined]
