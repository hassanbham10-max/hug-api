from transformers import pipeline

print("Loading News Topic Classifier...")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

TOPICS = ["Sports", "Technology", "Business", "Politics", "Health"]


def main():
    print("\nWelcome to News Topic Classifier!")
    print("Topics:", ", ".join(TOPICS))
    print("Type 'exit' to stop.\n")

    while True:
        headline = input("Enter a news headline: ").strip()

        if headline.lower() == "exit":
            print("Goodbye!")
            break

        if not headline:
            print("Please enter a headline.")
            continue

        try:
            result = classifier(
                headline,
                candidate_labels=TOPICS
            )

            print("\nHeadline:", headline)
            print("Best topic:", result["labels"][0])
            print(
                "Confidence:",
                f"{result['scores'][0] * 100:.2f}%"
            )

            print("\nTop 3 topics:")

            for i in range(min(3, len(result["labels"]))):
                print(
                    f"{i + 1}. {result['labels'][i]}: "
                    f"{result['scores'][i] * 100:.2f}%"
                )

            print()

    
        except Exception as error:
            print("Error:", error)


if __name__ == "__main__":
    main()