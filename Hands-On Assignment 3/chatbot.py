
"""
Terminal Q&A chatbot for MSCS-633 Hands-On Assignment 3.
Hands-On Assignment 3: Create a Simple Q&A Chatbot with Python
Name: Rabi Gurung

This script uses Python + ChatterBot and checks that Django is installed
to satisfy the assignment environment requirements.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import django
from chatterbot import ChatBot
from chatterbot.tagging import LowercaseTagger
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer

BOT_NAME = "AssignmentBot"
EXIT_COMMANDS = {"exit", "quit", "bye"}
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "chatbot_db.sqlite3"
TRAINING_MARKER_PATH = BASE_DIR / ".training_version"
TRAINING_VERSION = "2"
LOW_CONFIDENCE_REPLIES = [
    "Could you rephrase that question?",
    "I understand. Can you tell me a bit more?",
    "Interesting question. What specific part do you want to know?",
]
DIRECT_RESPONSES = {
    "hello": "Hello! How are you today?",
    "hi": "Hi there. How can I help you?",
    "good morning": "Good morning! I am doing very well, thank you for asking.",
    "how are you": "I am doing well, thank you for asking.",
    "what is your name": "My name is AssignmentBot.",
    "what can you do": "I can answer simple questions and chat with you in the terminal.",
    "thank you": "You are welcome.",
}
CUSTOM_TRAINING_DIALOG = [
    "hello",
    "Hello! How are you today?",
    "hi",
    "Hi there. How can I help you?",
    "how are you",
    "I am doing well, thank you for asking.",
    "what is your name",
    "My name is AssignmentBot.",
    "what can you do",
    "I can answer simple questions and chat with you in the terminal.",
    "thank you",
    "You are welcome.",
    "bye",
    "Goodbye.",
]


def parse_args() -> argparse.Namespace:
    """Parse optional CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the terminal Q&A chatbot.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain the bot even if a local SQLite database already exists.",
    )
    return parser.parse_args()


def build_chatbot() -> ChatBot:
    """Create a ChatterBot instance backed by a local SQLite database."""
    return ChatBot(
        BOT_NAME,
        storage_adapter="chatterbot.storage.SQLStorageAdapter",
        database_uri=f"sqlite:///{DATABASE_PATH.as_posix()}",
        # Use a lightweight tagger to avoid requiring external spaCy language models.
        tagger=LowercaseTagger,
        logic_adapters=[
            "chatterbot.logic.BestMatch",
            "chatterbot.logic.MathematicalEvaluation",
        ],
    )


def is_training_current() -> bool:
    """Return True when the current training marker matches the expected version."""
    if not TRAINING_MARKER_PATH.exists():
        return False
    return TRAINING_MARKER_PATH.read_text(encoding="utf-8").strip() == TRAINING_VERSION


def write_training_marker() -> None:
    """Persist the training version marker after successful training."""
    TRAINING_MARKER_PATH.write_text(TRAINING_VERSION, encoding="utf-8")


def train_chatbot(chatbot: ChatBot, retrain: bool = False) -> None:
    """Train the chatbot on first run or when retrain is requested."""
    should_train = retrain or not DATABASE_PATH.exists() or not is_training_current()

    if not should_train:
        return

    print("bot: Training in progress. This may take a minute...")
    trainer = ChatterBotCorpusTrainer(chatbot)
    trainer.train(
        "chatterbot.corpus.english.greetings",
        "chatterbot.corpus.english.conversations",
    )
    # Add a small curated dialog set for predictable assignment-style Q&A.
    ListTrainer(chatbot).train(CUSTOM_TRAINING_DIALOG)
    write_training_marker()
    print("bot: Training complete.")


def run_chat_loop(chatbot: ChatBot) -> None:
    """Start interactive terminal chat until the user exits."""
    print("bot: Hello! Ask me a question. Type 'exit' to quit.")

    while True:
        try:
            user_input = input("user: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbot: Goodbye!")
            break

        if not user_input:
            continue

        normalized = user_input.lower()

        if normalized in EXIT_COMMANDS:
            print("bot: Goodbye!")
            break

        if normalized in DIRECT_RESPONSES:
            print(f"bot: {DIRECT_RESPONSES[normalized]}")
            continue

        response = chatbot.get_response(user_input)
        if getattr(response, "confidence", 0.0) < 0.2:
            print(f"bot: {random.choice(LOW_CONFIDENCE_REPLIES)}")
            continue

        print(f"bot: {response}")


def main() -> None:
    """Entrypoint for terminal execution."""
    args = parse_args()

    # Environment check to confirm Django is installed for the assignment.
    print(f"bot: Environment ready (Django {django.get_version()}).")

    chatbot = build_chatbot()
    train_chatbot(chatbot, retrain=args.retrain)
    run_chat_loop(chatbot)


if __name__ == "__main__":
    main()
