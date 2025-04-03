import pandas
import time
import logging
import torch
import transformers
import liqfit.models
import liqfit.pipeline


def keywords_extraction(df: pandas.DataFrame,
                        model_id: str, verbose: bool,
                        batch_size: int = 8,
                        label_template: str = "{}") -> pandas.DataFrame:

    # Set up logger
    logger = logging.getLogger("KeywordExtraction")
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

    logger.info(f"Using model: {model_id}")

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Selected device: {device}")

    # Extract texts and class labels
    texts = df["text"].tolist()
    raw_labels = df.columns[1:].tolist()
    candidate_labels = [label_template.format(label) for label in raw_labels]

    # Load model & pipeline
    if model_id == "knowledgator/comprehend_it-multilingual-t5-base":
        model = liqfit.models.T5ForZeroShotClassification.from_pretrained(model_id).to(device)
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_id)
        classifier = liqfit.pipeline.ZeroShotClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            hypothesis_template=label_template,
            encoder_decoder=True,
            device=device
        )
    else:
        classifier = transformers.pipeline(
            "zero-shot-classification",
            model=model_id,
            tokenizer=transformers.AutoTokenizer.from_pretrained(model_id),
            device=0 if device == "cuda" else -1
        )

    # Initialize result storage
    results_dict = {
        "text": [],
        "time": []
    }
    for label in raw_labels:
        results_dict[label] = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        if verbose:
            logger.info(f"Processing batch {i}–{i + len(batch_texts) - 1}")

        start_time = time.perf_counter()
        outputs = classifier(batch_texts, candidate_labels=candidate_labels, multi_label=True)
        elapsed_time = time.perf_counter() - start_time

        if isinstance(outputs, dict):  # Happens when batch size = 1
            outputs = [outputs]

        for text, output in zip(batch_texts, outputs):
            results_dict["text"].append(text)
            results_dict["time"].append(elapsed_time / len(batch_texts))

            score_map = {label: 0.0 for label in candidate_labels}
            for label, score in zip(output["labels"], output["scores"]):
                score_map[label] = score

            for raw_label, templated_label in zip(raw_labels, candidate_labels):
                results_dict[raw_label].append(score_map[templated_label])

    return pandas.DataFrame(results_dict)
