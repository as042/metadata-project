from datetime import date
import os

from dataset import save_recent_studies, save_reconstructed_records
from dataset import filter_oa_studies
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
                  max_spend=None):
    """Harvest -> filter to open-access -> reconstruct, all four layers.

    Two of the four layers cost money. Measured on a 5-study / 52-record set:

    ==========================  =========  ==============================
    layer                       cost       what it needs
    ==========================  =========  ==============================
    1 direct                    free       already in the archive record
    2 ``harmonize``             free       a synonym table, no model
    3 ``from_text``             ~$0.0034   one model call per sample
    4 ``from_paper``            ~$0.013    one paper + one call per study
    ==========================  =========  ==============================

    So a run of this size is roughly **$0.18 for layer 3 plus $0.07 for layer
    4** — call it $0.25. Layer 3 scales with *records*; layer 4 scales with
    *studies*, and only those with an open-access paper (305 of 1,858 in the
    reference harvest), but each of its calls carries up to 30,000 characters of
    full text and is the largest single request the pipeline makes.

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

    ``create_dirs`` makes ``file_location`` if it does not exist, so a new run
    prefix needs no setup. With it off, a missing directory is reported **here**
    rather than after stage 1 has finished harvesting and tries to write — a
    crash at that point throws away every request the harvest just paid for.
    """
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
        **({} if max_spend is None else {"max_spend": max_spend}),
    )


if __name__ == "__main__":
    full_pipeline(file_location="./datasets/test3", max_studies=20)