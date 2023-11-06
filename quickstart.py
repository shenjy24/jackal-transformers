from datasets import load_dataset, Audio
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer


def zero_shot_classification():
    classifier = pipeline("zero-shot-classification")
    classifier(
        "This is a course about the Transformers library",
        candidate_labels=["education", "politics", "business"],
    )


def sentiment_analysis(text):
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    res = classifier(text)
    print(res)


def automatic_speech_recognition():
    speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
    # 加载音频数据集
    dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
    # 采样率匹配
    dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
    result = speech_recognizer(dataset[:4]["audio"])
    print([d["text"] for d in result])


if __name__ == "__main__":
    sentiment_analysis("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
