import argparse
import os
import json
import pandas
import logging
import utilities


def get_localized_template(template_entry, language):
    if not template_entry.get("localization", False):
        return template_entry["template"]
    return template_entry.get(language, template_entry["template"])  # fallback


def infer_label_chunk_size(df: pandas.DataFrame) -> int:
    """
    Conservative default for decoder-only; can be overridden by args.
    """
    num_labels = max(0, len(df.columns) - 1)
    if num_labels <= 32:
        return num_labels
    return 32


def infer_dataset_kind(dataset_id: str) -> str:
    """
    Your naming convention:
      - quotes_c_* => concepts (19)
      - quotes_k_* => keywords (256)
    """
    dataset_id = (dataset_id or "").lower()
    if "_c_" in dataset_id:
        return "concepts"
    if "_k_" in dataset_id:
        return "keywords"
    return "unknown"


def is_language_compatible(model_lang: str, dataset_lang: str) -> bool:
    model_lang = (model_lang or "").upper().strip()
    dataset_lang = (dataset_lang or "").upper().strip()

    if model_lang in ["MULTI", "ANY", "ALL", "X"]:
        return True
    return model_lang == dataset_lang


def decoder_defaults_for_h100(dataset_kind: str) -> dict:
    """
    Recommended starting points for H100 (94GB) given your text lengths.
    These are only applied when the user does NOT explicitly override via args.
    """
    if dataset_kind == "concepts":
        return {
            "batch_size": 64,
            "label_chunk_size": 19,
            "prompt_batch_size": 1024,
            "max_length": 512
        }
    if dataset_kind == "keywords":
        return {
            "batch_size": 8,
            "label_chunk_size": 128,
            "prompt_batch_size": 512,
            "max_length": 512
        }
    # Fallback
    return {
        "batch_size": 8,
        "label_chunk_size": 32,
        "prompt_batch_size": 256,
        "max_length": 512
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation of datasets and models (NLI templates; decoder-only single run with H100-friendly knobs)."
    )

    parser.add_argument("--models_csv", required=True, type=str,
                        help="CSV with columns: ID, LANGUAGE, MODEL, optional TYPE (NLI|DECODER)")
    parser.add_argument("--datasets_csv", required=True, type=str,
                        help="CSV with columns: ID, LANGUAGE, PATH")
    parser.add_argument("--output_folder", required=True, type=str,
                        help="Directory to save output CSVs")
    parser.add_argument("--templates_json", required=True, type=str,
                        help="JSON file with label templates (and optional localization) - applied only to NLI models")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    # Shared
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for inference (texts per batch). If omitted, decoder-only uses dataset-based defaults.")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Max token length for decoder-only prompt truncation. If omitted, decoder-only uses defaults.")

    # Decoder-only specific
    parser.add_argument("--decoder_label_template", type=str, default="{}",
                        help="Label formatting for decoder-only models (default: identity)")
    parser.add_argument("--label_chunk_size", type=int, default=None,
                        help="Decoder-only: number of labels per chunk. If omitted, uses dataset-based defaults.")
    parser.add_argument("--prompt_batch_size", type=int, default=None,
                        help="Decoder-only: micro-batch size for prompts inside each label chunk. If omitted, uses dataset-based defaults.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    models_df = pandas.read_csv(args.models_csv)
    datasets_df = pandas.read_csv(args.datasets_csv)

    with open(args.templates_json, "r", encoding="utf-8") as f:
        templates = json.load(f)

    os.makedirs(args.output_folder, exist_ok=True)

    if "TYPE" not in models_df.columns:
        models_df["TYPE"] = "NLI"
    models_df["TYPE"] = models_df["TYPE"].fillna("NLI").astype(str).str.upper()

    for _, dataset_row in datasets_df.iterrows():
        data_id = dataset_row["ID"]
        data_path = dataset_row["PATH"]
        dataset_lang = str(dataset_row["LANGUAGE"]).upper()

        df = pandas.read_csv(data_path)

        dataset_kind = infer_dataset_kind(data_id)
        h100_defaults = decoder_defaults_for_h100(dataset_kind)

        # If you want your old behaviour for NLI batch size, keep a default:
        nli_batch_size = args.batch_size if args.batch_size is not None else 8

        for _, model_row in models_df.iterrows():
            model_id = model_row["ID"]
            model_hf_id = model_row["MODEL"]
            model_type = str(model_row["TYPE"]).upper()
            model_lang = str(model_row["LANGUAGE"]).upper()

            if not is_language_compatible(model_lang, dataset_lang):
                if args.verbose:
                    logging.info(
                        f"Skipping model {model_id} (LANG={model_lang}) on dataset {data_id} (LANG={dataset_lang})"
                    )
                continue

            if model_type == "DECODER":
                # Decoder-only: ignore templates_json and run once
                label_template = args.decoder_label_template

                # Use dataset-based defaults unless overridden
                decoder_batch_size = args.batch_size if args.batch_size is not None else h100_defaults["batch_size"]
                decoder_label_chunk_size = args.label_chunk_size if args.label_chunk_size is not None else h100_defaults["label_chunk_size"]
                decoder_prompt_batch_size = args.prompt_batch_size if args.prompt_batch_size is not None else h100_defaults["prompt_batch_size"]
                decoder_max_length = args.max_length if args.max_length is not None else h100_defaults["max_length"]

                print(
                    f"\n🧪 Dataset: '{data_id}' ({dataset_kind}) | Model: '{model_id}' | Type: DECODER | "
                    f"ModelLang: {model_lang} | DataLang: {dataset_lang} | "
                    f"batch_size={decoder_batch_size} label_chunk_size={decoder_label_chunk_size} "
                    f"prompt_batch_size={decoder_prompt_batch_size} max_length={decoder_max_length}"
                )

                results_df = utilities.decoder_only_keywords_extraction(
                    df=df,
                    model_id=model_hf_id,
                    verbose=args.verbose,
                    batch_size=decoder_batch_size,
                    label_template=label_template,
                    label_chunk_size=decoder_label_chunk_size,
                    max_length=decoder_max_length,
                    prompt_language=dataset_lang,
                    prompt_batch_size=decoder_prompt_batch_size
                )

                # No template_id for decoder-only
                output_filename = f"{data_id}_{model_id}.csv"
                output_path = os.path.join(args.output_folder, output_filename)
                results_df.to_csv(output_path, index=False)
                print(f"✅ Saved: {output_path}")

            else:
                # NLI: run for each label template
                for template_entry in templates:
                    template_id = template_entry.get("ID")
                    label_template = get_localized_template(template_entry, dataset_lang)

                    print(
                        f"\n🧪 Dataset: '{data_id}' | Model: '{model_id}' | Type: NLI | "
                        f"ModelLang: {model_lang} | DataLang: {dataset_lang} | "
                        f"batch_size={nli_batch_size} | TemplateID: {template_id} | Template: '{label_template}'"
                    )

                    results_df = utilities.keywords_extraction(
                        df=df,
                        model_id=model_hf_id,
                        verbose=args.verbose,
                        batch_size=nli_batch_size,
                        label_template=label_template
                    )

                    output_filename = f"{data_id}_{model_id}_{template_id}.csv"
                    output_path = os.path.join(args.output_folder, output_filename)
                    results_df.to_csv(output_path, index=False)
                    print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    main()
