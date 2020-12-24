import os
import time

import requests
import io
import hashlib

from PIL import Image
from selenium import webdriver

import signal

from glob import glob

##############
# Parameters
##############

number_of_images = 400
GET_IMAGE_TIMEOUT = 2
SLEEP_BETWEEN_INTERACTIONS = 0.1
SLEEP_BEFORE_MORE = 5

output_path = "/home/ladvien/deep_arcane/images/0_raw/0_scraped/"

# Get terms already recorded.
dirs = glob(output_path + "*")
dirs = [dir.split("/")[-1].replace("_", " ") for dir in dirs]

search_terms = [
    "black and white magic symbol icon",
    "black and white arcane symbol icon",
    "black and white mystical symbol",
    "black and white useful magic symbols icon",
    "black and white ancient magic sybol icon",
    "black and white key of solomn symbol icon",
    "black and white historic magic symbol icon",
    "black and white symbols of demons icon",
    "black and white magic symbols from book of enoch",
    "black and white historical magic symbols icons",
    "black and white witchcraft magic symbols icons",
    "black and white occult symbols icons",
    "black and white rare magic occult symbols icons",
    "black and white rare medieval occult symbols icons",
    "black and white alchemical symbols icons",
    "black and white demonology symbols icons",
    "black and white magic language symbols icon",
    "black and white magic words symbols glyphs",
    "black and white sorcerer symbols",
    "black and white magic symbols of power",
    "occult religious symbols from old books",
    "conjuring symbols",
    "magic wards",
    "esoteric magic symbols",
    "demon summing symbols",
    "demon banishing symbols",
    "esoteric magic sigils",
    "esoteric occult sigils",
    "ancient cult symbols",
    "gypsy occult symbols",
    "Feri Tradition symbols",
    "Quimbanda symbols",
    "Nagualism symbols",
    "Pow-wowing symbols",
    "Onmyodo symbols",
    "Ku magical symbols",
    "Seidhr And Galdr magical symbols",
    "Greco-Roman magic symbols",
    "Levant magic symbols",
    "Book of the Dead magic symbols",
    "kali magic symbols",
]

# Exclude terms already stored.
search_terms = [term for term in search_terms if term not in dirs]

##########
# Scrap
##########

wd = webdriver.Chrome()
wd.get("https://google.com")


# Credit:
# https://stackoverflow.com/a/22348885
class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def fetch_image_urls(
    query: str,
    max_links_to_fetch: int,
    wd: webdriver,
    sleep_between_interactions: int = 1,
):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        print(
            f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}"
        )

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls
            actual_images = wd.find_elements_by_css_selector("img.n3VNCb")
            for actual_image in actual_images:
                if actual_image.get_attribute(
                    "src"
                ) and "http" in actual_image.get_attribute("src"):
                    image_urls.add(actual_image.get_attribute("src"))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(SLEEP_BEFORE_MORE)

            not_what_you_want_button = ""
            try:
                not_what_you_want_button = wd.find_element_by_css_selector(".r0zKGf")
            except:
                pass

            # If there are no more images return.
            if not_what_you_want_button:
                print("No more images available.")
                return image_urls

            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button and not not_what_you_want_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls


def persist_image(folder_path: str, url: str):
    try:
        print("Getting image")
        with timeout(GET_IMAGE_TIMEOUT):
            image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert("RGB")
        file_path = os.path.join(
            folder_path, hashlib.sha1(image_content).hexdigest()[:10] + ".jpg"
        )
        with open(file_path, "wb") as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")


def search_and_download(search_term: str, target_path="./images", number_images=5):
    target_folder = os.path.join(target_path, "_".join(search_term.lower().split(" ")))

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with webdriver.Chrome() as wd:
        res = fetch_image_urls(
            search_term,
            number_images,
            wd=wd,
            sleep_between_interactions=SLEEP_BETWEEN_INTERACTIONS,
        )

        if res is not None:
            for elem in res:
                persist_image(target_folder, elem)
        else:
            print(f"Failed to return links for term: {search_term}")


for term in search_terms:
    search_and_download(term, output_path, number_of_images)
