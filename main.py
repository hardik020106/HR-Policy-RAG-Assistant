from hr_assistant.pipeline import ask, build_hr_assistant

def main():
    # Build the HR assistant
    hr_agent = build_hr_assistant()

    # Example question to ask the HR agent
    question = "What are the company's leave policies?"
    response = ask(hr_agent, question)
    print(f"Question: {question}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()