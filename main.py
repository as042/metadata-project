from dataset import save_recent_studies
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


if __name__ == "__main__":
    set_entrez_credentials(
        api_key=read_credential("api_key.txt"),
        email=read_credential("email.txt"),
    )

    save_recent_studies("./datasets/recent_studies.json", 200)
    filter_oa_studies(
        in_path="./datasets/recent_studies.json",
        out_path="./datasets/oa_studies.json",
    )