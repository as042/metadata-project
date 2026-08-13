from datetime import date
import os

import claude
import reconstruct
from dataset import save_recent_studies, save_reconstructed_records
from dataset import filter_oa_studies, validate_model_settings
from project import set_entrez_credentials


def read_credential(path):
    """Read a one-line credential file.

    Stripped because a trailing newline is invisible in an editor but fatal: NCBI
    answers a key with whitespace in it with HTTP 400 "API key invalid", and 400
    isn't retried, so the first request of a run would abort it.
    """
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()


def full_pipeline(file_location="./datasets/test", dataset_prefix="test", max_studies=10,
                  harmonize=True, from_text=True, from_paper=True, create_dirs=True,
                  max_spend=None, claude_key_file=None,
                  text_model=None, text_effort=None, text_thinking=None,
                  paper_model=None, paper_effort=None, paper_thinking=None):
    """Harvest -> filter to open-access -> reconstruct, all four layers.

    Two of the four layers cost money. Measured on a 5-study / 52-record set:

    ==========================  =========  ==============================
    layer                       cost       what it needs
    ==========================  =========  ==============================
    1 direct                    free       already in the archive record
    2 ``harmonize``             free       a synonym table, no model
    3 ``from_text``             ~$0.0035   one model call per sample
    4 ``from_paper``            ~$0.013    one paper + one call per study
    ==========================  =========  ==============================

    So a run of this size is roughly **$0.18 for layer 3 plus $0.07 for layer
    4** — call it $0.25. Layer 3 scales with *records*; layer 4 scales with
    *studies*, and only those with an open-access paper (305 of 1,858 in the
    reference harvest), but each of its calls carries up to 30,000 characters of
    full text and is the largest single request the pipeline makes.

    **Those two figures are for the default model (Haiku 4.5) with no thinking.**
    They scale with whatever ``text_model`` / ``paper_model`` select — measured
    on this same set, Sonnet 5 without thinking came to $0.49 and Sonnet 5 with
    thinking to $1.25. The printed estimate does the scaling for you; every run
    also prints what it actually billed.

    Pass ``from_text=False, from_paper=False`` for a free run that stops at what
    SRA states outright, or ``from_paper=False`` to skip only the paper layer.

    Neither layer batches by default (``reconstruct.TEXT_BATCH`` is off), so
    calls go out one at a time: progress is visible and a Ctrl-C stops the spend
    where it stands. Turning batching on saves 12-28% and gives that up.

    ``max_spend`` caps the reconstruction stage: the estimate is printed and the
    run refuses to start above the limit, having spent nothing. ``None`` here
    means "use dataset.MAX_SPEND" ($1.00) rather than "unlimited" — pass a
    number to authorise a bigger run. A 20-study harvest that yielded only two
    open-access studies cost $7 because one of them held 1,664 experiments, and
    nothing connected those two facts before it spent.

    ``claude_key_file`` names which Anthropic credential pays for layers 3 and 4,
    and is **required whenever either is on** — there is no default key file, so
    a run that names none is refused rather than quietly billing whatever
    credential happens to sit in the repo. The check happens **here**, before the
    harvest, so a missing or malformed key costs a local file read rather than
    surfacing after stage 1 has spent its requests. Reading the file spends
    nothing: the key is not used until layer 3 makes its first call.

    Pass ``from_text=False, from_paper=False`` and no key at all for a free run.
    Naming a fresh key does not raise ``max_spend``: the cap is a per-run
    estimate check, not a balance, and it still applies unchanged.

    ``text_model`` / ``text_effort`` / ``text_thinking`` pick what layer 3 runs
    on, and the ``paper_*`` trio does the same for layer 4 — they are separate
    because the layers are: layer 3 asks a small question tens of thousands of
    times, while layer 4 makes one large call per open-access study. Paying for a
    better model on layer 4 alone is cheap; paying for it on layer 3 is not.
    ``None`` keeps :mod:`reconstruct`'s defaults (Haiku 4.5, no effort, no
    thinking).

    **The cost estimate follows these settings**, so ``max_spend`` keeps
    protecting the run after you change them — the per-unit figures are measured
    on Haiku 4.5 and scaled by price and thinking for whatever you select. The
    thinking multipliers are estimates rather than measurements and are rounded
    up: expect the guard to refuse slightly too early rather than too late.
    Combinations the API rejects outright (effort on Haiku 4.5, thinking off on
    Opus 5 at ``xhigh``) are refused **here**, before the harvest, because a 400
    partway through a run leaves everything before it already billed.

    ``create_dirs`` makes ``file_location`` if it does not exist, so a new run
    prefix needs no setup. With it off, a missing directory is reported **here**
    rather than after stage 1 has finished harvesting and tries to write — a
    crash at that point throws away every request the harvest just paid for.
    """
    # Credential first: it is the cheapest thing to get wrong and the most
    # expensive to discover late, so it is checked before a directory is made,
    # before NCBI is contacted, and before a single study is harvested.
    if claude_key_file is not None:
        claude.set_api_key(path=claude_key_file)
    if from_text or from_paper:
        claude.require_api_key()

    # Same reasoning for the model settings: a rejected combination costs a
    # local check here, or a 400 after the harvest and part of layer 3 have run.
    reconstruct.configure_models(
        text_model=text_model, text_effort=text_effort, text_thinking=text_thinking,
        paper_model=paper_model, paper_effort=paper_effort, paper_thinking=paper_thinking,
    )
    validate_model_settings(
        reconstruct.TEXT_MODEL, reconstruct.TEXT_EFFORT, reconstruct.TEXT_THINKING,
        reconstruct.PAPER_MODEL, reconstruct.PAPER_EFFORT, reconstruct.PAPER_THINKING,
        from_text=from_text, from_paper=from_paper,
    )

    if create_dirs:
        os.makedirs(file_location, exist_ok=True)
    elif not os.path.isdir(file_location):
        raise FileNotFoundError(
            f"{file_location} does not exist. Create it, or pass create_dirs=True."
        )

    set_entrez_credentials(
        api_key=read_credential("api_key.txt"),
        email=read_credential("email.txt"),
    )

    save_recent_studies(f"{file_location}/{dataset_prefix}_studies.json", max_studies, after_date=date(2016, 1, 1))

    filter_oa_studies(
        in_path=f"{file_location}/{dataset_prefix}_studies.json",
        out_path=f"{file_location}/{dataset_prefix}_filtered.json",
    )

    save_reconstructed_records(
        in_path=f"{file_location}/{dataset_prefix}_filtered.json",
        out_path=f"{file_location}/{dataset_prefix}_reconstructed.json",
        harmonize=harmonize,
        from_text=from_text,
        from_paper=from_paper,
        text_model=text_model, text_effort=text_effort, text_thinking=text_thinking,
        paper_model=paper_model, paper_effort=paper_effort, paper_thinking=paper_thinking,
        **({} if max_spend is None else {"max_spend": max_spend}),
    )


if __name__ == "__main__":
    # full_pipeline(file_location="./datasets/test3", max_studies=20)

    save_reconstructed_records(
        in_path="datasets/test2/test_filtered.json",
        out_path="datasets/test2/test_reconstructed5.json",
        harmonize=True,
        from_text=True,
        text_model=claude.SONNET_5,
        from_paper=True,
        paper_model=claude.SONNET_5,
        max_spend=1.0,
        claude_key_file="anton_claude_api_key.txt"
    )