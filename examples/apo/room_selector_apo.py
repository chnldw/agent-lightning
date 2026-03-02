# Copyright (c) Microsoft. All rights reserved.

"""This sample code demonstrates how to use an existing APO algorithm to tune the prompts."""

import difflib
import logging
from typing import Tuple, cast

from openai import AsyncOpenAI
from room_selector import RoomSelectionTask, load_room_tasks, prompt_template_baseline, room_selector

from agentlightning import Trainer, setup_logging
from agentlightning.adapter import TraceToMessages
from agentlightning.algorithm.apo import APO
from agentlightning.types import Dataset


def load_train_val_dataset() -> Tuple[Dataset[RoomSelectionTask], Dataset[RoomSelectionTask]]:
    dataset_full = load_room_tasks()
    train_split = len(dataset_full) // 2
    dataset_train = [dataset_full[i] for i in range(train_split)]
    dataset_val = [dataset_full[i] for i in range(train_split, len(dataset_full))]
    return cast(Dataset[RoomSelectionTask], dataset_train), cast(Dataset[RoomSelectionTask], dataset_val)


def setup_apo_logger(file_path: str = "apo.log") -> None:
    """Dump a copy of all the logs produced by APO algorithm to a file."""

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger("agentlightning.algorithm.apo").addHandler(file_handler)


def print_training_results(algo: APO[RoomSelectionTask]) -> None:
    """Print all prompt variants, their scores, and a diff analysis after training."""
    history = algo.get_prompt_history()
    if not history:
        print("No prompt variants found in history.")
        return

    seed = next((p for p in history if p.version == "v0"), None)
    best_version = algo._history_best_version
    best_score = algo._history_best_score

    # --- Ranked table of all variants ---
    print("\n" + "=" * 90)
    print("  ALL PROMPT VARIANTS (ranked by validation score)")
    print("=" * 90)
    print(f"  {'Rank':<6} {'Version':<10} {'Score':<12} {'Prompt Preview'}")
    print("  " + "-" * 86)
    for rank, prompt in enumerate(history, 1):
        preview = prompt.prompt_template.template[:60].replace("\n", "\\n")
        score_str = f"{prompt.score:.4f}" if prompt.score is not None else "N/A"
        marker = "  <-- BEST" if prompt.version == best_version else ""
        print(f"  {rank:<6} {prompt.version:<10} {score_str:<12} {preview}...{marker}")

    # --- Score improvement summary ---
    if seed and seed.score is not None and best_score > float("-inf"):
        improvement = best_score - seed.score
        pct = (improvement / abs(seed.score) * 100) if seed.score != 0 else 0.0
        print(f"\n  Seed score:  {seed.score:.4f} ({seed.version})")
        print(f"  Best score:  {best_score:.4f} ({best_version}, full validation)")
        print(f"  Improvement: {improvement:+.4f} ({pct:+.1f}%)")

    # --- Best prompt (full text) ---
    print(f"\n{'=' * 90}")
    print(f"  BEST PROMPT: {best_version} (full-validation score: {best_score:.4f})")
    print("=" * 90)
    print(algo.get_best_prompt().template)

    # --- Diff: seed vs best ---
    if seed and best_version and best_version != seed.version:
        best_prompt = next((p for p in history if p.version == best_version), None)
        if best_prompt:
            print(f"\n{'=' * 90}")
            print(f"  DIFF ANALYSIS: {seed.version} (seed) -> {best_version} (best)")
            print("=" * 90)
            seed_lines = seed.prompt_template.template.splitlines(keepends=True)
            best_lines = best_prompt.prompt_template.template.splitlines(keepends=True)
            diff = list(
                difflib.unified_diff(
                    seed_lines,
                    best_lines,
                    fromfile=f"seed ({seed.version}, score={seed.score})",
                    tofile=f"best ({best_version}, score={best_score:.4f})",
                )
            )
            if diff:
                print("".join(diff))
            else:
                print("  (identical — best prompt is the seed)")

    # --- Full text of each variant for side-by-side comparison ---
    if len(history) > 1:
        print(f"\n{'=' * 90}")
        print("  FULL PROMPT TEXT FOR EACH VARIANT")
        print("=" * 90)
        for prompt in history:
            score_str = f"{prompt.score:.4f}" if prompt.score is not None else "N/A"
            marker = "  <-- BEST" if prompt.version == best_version else ""
            print(f"\n  --- {prompt.version} (score: {score_str}){marker} ---")
            print(prompt.prompt_template.template)


def main() -> None:
    setup_logging()
    setup_apo_logger()

    import os

    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))

    algo = APO[RoomSelectionTask](
        openai_client,
        gradient_model="gpt-5.2",
        apply_edit_model="gpt-5-mini",
        val_batch_size=16,
        gradient_batch_size=4,
        beam_width=4,
        branch_factor=4,
        beam_rounds=3,
        _poml_trace=True,
    )
    trainer = Trainer(
        algorithm=algo,
        # Increase the number of runners to run more rollouts in parallel
        n_runners=8,
        # APO algorithm needs a baseline
        # Set it either here or in the algo
        initial_resources={
            # The resource key can be arbitrary
            "prompt_template": prompt_template_baseline()
        },
        # APO algorithm needs an adapter to process the traces produced by rollouts
        # Use this adapter to convert spans to messages
        adapter=TraceToMessages(),
    )
    dataset_train, dataset_val = load_train_val_dataset()
    trainer.fit(agent=room_selector, train_dataset=dataset_train, val_dataset=dataset_val)

    print_training_results(algo)

    print("\n" + "=" * 90)
    print("  BEST PROMPT TEMPLATE")
    print("=" * 90)
    print(algo.get_best_prompt().template)
    print("=" * 90)


if __name__ == "__main__":
    main()
