from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib

import PIL.Image

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent


def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": get_datasets_user_agent()},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image


def fetch_images(batch, num_threads, timeout=None, retries=0):
    fetch_single_image_with_args = partial(
        fetch_single_image, timeout=timeout, retries=retries
    )
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(
            executor.map(fetch_single_image_with_args, batch["image_url"])
        )
    return batch


num_threads = 20
dset = load_dataset(
    "conceptual_captions",
    split="train[10000:20000]",
    cache_dir="./data/ConceptualCaptions",
)


dset = dset.filter(lambda example: len(example["caption"]) < 75)

dset = dset.map(
    fetch_images,
    batched=True,
    batch_size=100,
    cache_file_name="next_processed_cache",
    fn_kwargs={"num_threads": num_threads},
)


dset = dset.remove_columns("image_url")
dset = dset.filter(lambda example: example["image"] is not None)

dset.save_to_disk("./data/next_processed")
