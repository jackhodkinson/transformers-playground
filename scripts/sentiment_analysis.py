from transformers import pipeline


def main():
    classifier = pipeline("sentiment-analysis")
    result = classifier("We are very happy to show you the 🤗 Transformers library.")
    print(result)


if __name__ == "__main__":
    main()
