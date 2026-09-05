from hr_assistant.pipeline import ask, build_hr_assistant
from hr_assistant.logger import get_logger

logger = get_logger(__name__)
def main():
    # Build the HR assistant
    hr_agent = build_hr_assistant()

    # Example question to ask the HR agent
    question = "What are the company's leave policies?"
    response = ask(hr_agent, question)
    logger.info(f"Question: {question}")
    logger.info(f"Response: {response}")

if __name__ == "__main__":
    main()