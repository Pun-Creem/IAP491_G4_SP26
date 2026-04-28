import os
import sys
import platform
import logging
from datetime import datetime
import gensim
import numpy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tkinter import Tk
from tkinter.filedialog import askdirectory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "malware_doc2vec.model")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "train_model.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def log_environment():

    logger.info("---------- EXPERIMENT SETUP ----------")
    logger.info(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Gensim version: {gensim.__version__}")
    logger.info(f"Numpy version: {numpy.__version__}")
    logger.info("--------------------------------------")


def load_patterns_from_folder(folder):

    documents = []
    empty_files = []

    for file in os.listdir(folder):

        if file.endswith(".txt"):

            path = os.path.join(folder, file)

            tokens = []

            with open(path, "r", encoding="utf-8") as f:

                for line in f:

                    token = line.strip()

                    if token != "":
                        tokens.append(token)

            tag = os.path.splitext(file)[0]

            if len(tokens) > 0:

                documents.append(
                    TaggedDocument(tokens, [tag])
                )

            else:
                empty_files.append(file)

    if len(empty_files) > 0:
        logger.warning(f"Empty files skipped ({len(empty_files)}): {empty_files}")

    return documents


def train_model(folder):

    start_time = datetime.now()

    logger.info("========== TRAINING STARTED ==========")
    log_environment()
    logger.info(f"Start time: {start_time}")
    logger.info(f"Dataset folder: {folder}")

    documents = load_patterns_from_folder(folder)

    logger.info(f"Documents loaded: {len(documents)}")

    model = Doc2Vec(
        dm=0,
        vector_size=100,
        window=10,
        min_count=1,
        workers=1,
        epochs=200,
        seed=42
    )

    logger.info("Model config: dm=0, vector_size=100, window=10, min_count=1, workers=1, epochs=200, seed=42")

    logger.info("Building vocabulary...")
    model.build_vocab(documents)

    logger.info(f"Vocabulary size: {len(model.wv)}")

    logger.info("Training Doc2Vec model...")

    model.train(
        documents,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )

    model.save(MODEL_FILE)

    end_time = datetime.now()
    duration = end_time - start_time

    logger.info(f"Model saved at: {MODEL_FILE}")
    logger.info(f"End time: {end_time}")
    logger.info(f"Total training time: {duration}")
    logger.info("========== TRAINING COMPLETED ==========")


if __name__ == "__main__":

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    logger.info("Select folder containing malware pattern files")

    folder = askdirectory(parent=root)

    root.destroy()

    if not folder:
        logger.warning("No folder selected. Exiting.")
    else:
        train_model(folder)