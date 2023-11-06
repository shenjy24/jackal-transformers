from transformers import pipeline


def automatic_speech_recognition():
    transcriber = pipeline(task="automatic-speech-recognition")
    res = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
    print(res)


def vision_pipeline():
    vision_classifier = pipeline(model="google/vit-base-patch16-224")
    preds = vision_classifier(
        images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
    preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
    print(preds)


def multimodal():
    vqa = pipeline(model="impira/layoutlm-document-qa")
    res = vqa(
        image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
        question="What is the invoice number?"
    )
    print(res)


if __name__ == "__main__":
    multimodal()
