from datasets import load_dataset, Audio
from transformers import AutoTokenizer, AutoFeatureExtractor


def text_preprocess():
    """
    文本处理
    :return:
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
    print(encoded_input)

    text = tokenizer.decode(encoded_input["input_ids"])
    print(text)


def audio_preprocess():
    dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
    print(dataset[0]["audio"])
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    print(dataset[0]["audio"])
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    audio_input = [dataset[0]["audio"]["array"]]
    features = feature_extractor(audio_input, sampling_rate=16000)
    print(features)
    print(dataset[0]["audio"]["array"].shape)
    print(dataset[1]["audio"]["array"].shape)


def audio_preprocess2():
    dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    print(dataset[0]["audio"])
    processed_dataset = preprocess_function(dataset[:5])
    print(dataset[0]["audio"])
    print(processed_dataset["input_values"][0].shape)


def preprocess_function(examples):
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        padding=True,
        max_length=100000,
        truncation=True,
    )
    return inputs


def image_preprocess():
    """
    图片预处理
    :return:
    """
    dataset = load_dataset("food101", split="train[:100]")
    print(dataset[0])


if __name__ == "__main__":
    image_preprocess()
