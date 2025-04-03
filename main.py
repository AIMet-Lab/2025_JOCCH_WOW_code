import argparse
import os
import json
import pandas
import utilities
import logging


def get_localized_template(template_entry, language):
    if not template_entry.get("localization", False):
        return template_entry["template"]
    return template_entry.get(language, template_entry["template"])  # fallback


def main():
    parser = argparse.ArgumentParser(description="Run evaluation of datasets and models with label templates.")

    parser.add_argument("--models_csv", required=True, type=str,
                        help="CSV with columns: ID, LANGUAGE ('EN' or 'IT'), MODEL")
    parser.add_argument("--datasets_csv", required=True, type=str,
                        help="CSV with columns: ID, LANGUAGE ('EN' or 'IT'), PATH")
    parser.add_argument("--output_folder", required=True, type=str,
                        help="Directory to save output CSVs")
    parser.add_argument("--templates_json", required=True, type=str,
                        help="JSON file with label templates (and optional localization)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    models_df = pandas.read_csv(args.models_csv)
    datasets_df = pandas.read_csv(args.datasets_csv)
    with open(args.templates_json, "r", encoding="utf-8") as f:
        templates = json.load(f)

    os.makedirs(args.output_folder, exist_ok=True)

    for _, dataset_row in datasets_df.iterrows():
        data_id = dataset_row["ID"]
        data_path = dataset_row["PATH"]
        dataset_lang = dataset_row["LANGUAGE"]
        df = pandas.read_csv(data_path)

        for _, model_row in models_df.iterrows():
            model_id = model_row["ID"]
            model_hf_id = model_row["MODEL"]
            model_lang = model_row["LANGUAGE"]

            if model_lang != dataset_lang:
                continue  # skip non-matching language combinations

            for template_entry in templates:

                label_template = get_localized_template(template_entry, dataset_lang)

                print(f"\n🧪 Dataset: '{data_id}' | Model: '{model_id}' | Lang: {dataset_lang} | Template: '{label_template}'")

                results_df = utilities.keywords_extraction(
                    df,
                    model_id=model_hf_id,
                    verbose=args.verbose,
                    batch_size=args.batch_size,
                    label_template=label_template
                )

                # Safe template ID for filename
                template_id = template_entry.get("ID")
                output_filename = f"{data_id}_{model_id}_{template_id}.csv"
                output_path = os.path.join(args.output_folder, output_filename)

                results_df.to_csv(output_path, index=False)
                print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    main()
