#!/usr/bin/env python3
"""Benchmark and train production valence classifiers from LLM-labeled examples.

The benchmark holds out complete posts, not individual post-frame applications,
so the same text cannot occur in both a training and evaluation fold. Sentence
embedding models are included when sentence-transformers is installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder


DEFAULT_LABELS = "derived/valence/valence_sample_v2_n10000_seed42.csv"
DEFAULT_OUT_DIR = "derived/valence/models"
DEFAULT_PARQUET = "derived/analysis_posts_clean.parquet"
TARGETS = {
    "focal_valence": {"excluded_labels": {"Not Applicable"}},
}
STRUCTURED_COLUMNS = [
    "source_country",
    "account_type",
    "frame_domain",
    "frame_subframe",
    "focal_actor",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Labeled training CSV")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for metrics and models")
    parser.add_argument("--seed", type=int, default=42, help="Grouped cross-validation seed")
    parser.add_argument("--folds", type=int, default=5, help="Number of grouped CV folds")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformers model used when the optional dependency is installed",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=64, help="Sentence-embedding batch size")
    parser.add_argument(
        "--include-tfidf",
        action="store_true",
        help="Also benchmark the legacy TF-IDF baselines; embeddings are used by default",
    )
    parser.add_argument(
        "--predict-rest",
        action="store_true",
        help="Apply the selected production models to unlabeled substantive post-frame applications",
    )
    parser.add_argument("--parquet", default=DEFAULT_PARQUET, help="Clean Parquet corpus for --predict-rest")
    parser.add_argument(
        "--prediction-out",
        default="derived/valence/valence_predictions_rest.parquet",
        help="Prediction Parquet for --predict-rest",
    )
    parser.add_argument("--batch-size", type=int, default=5000, help="Prediction batch size")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast 600-row, two-fold grouped test and use a separate output directory",
    )
    return parser


def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["text"] = data["text"].fillna("").astype(str)
    for column in STRUCTURED_COLUMNS:
        data[column] = data[column].fillna("None").astype(str).str.strip().replace("", "None")
    data["group_id"] = data["merge_key"].astype(str) + ":" + data["key_occurrence"].astype(str)
    # Keep all known structured context available to the production classifier.
    data["model_text"] = (
        "source_country=" + data["source_country"]
        + " account_type=" + data["account_type"]
        + " frame=" + data["frame_domain"]
        + " subframe=" + data["frame_subframe"]
        + " focal_actor=" + data["focal_actor"]
        + " text=" + data["text"]
    )
    return data


def build_tfidf_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def fit_structured_encoder(data: pd.DataFrame) -> tuple[OneHotEncoder, object]:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    return encoder, encoder.fit_transform(data[STRUCTURED_COLUMNS])


def combine_embedding_and_structure(
    embeddings: np.ndarray,
    structured_features: object,
    dense: bool,
) -> object:
    if dense:
        return np.hstack(
            [
                embeddings.astype(np.float32, copy=False),
                structured_features.toarray().astype(np.float32, copy=False),
            ]
        )
    return hstack([csr_matrix(embeddings), structured_features], format="csr")


def classifier_factories() -> dict[str, object]:
    return {
        "majority_baseline": lambda: DummyClassifier(strategy="most_frequent"),
        "tfidf_logistic_regression": lambda: LogisticRegression(
            max_iter=1_500, class_weight="balanced", C=1.0
        ),
        "tfidf_linear_svm": lambda: LinearSVC(class_weight="balanced", C=1.0),
        "embedding_logistic_regression": lambda: LogisticRegression(
            max_iter=1_500, class_weight="balanced", C=1.0
        ),
        "embedding_linear_svm": lambda: LinearSVC(class_weight="balanced", C=1.0),
        "embedding_mlp": lambda: MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=12,
            max_iter=200,
            random_state=42,
        ),
    }


def metric_rows(y_true: pd.Series, y_pred: np.ndarray, target: str, model: str, fold: int) -> list[dict[str, object]]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    rows = [
        {
            "target": target,
            "model": model,
            "fold": fold,
            "metric": "accuracy",
            "value": accuracy,
            "label": "all",
            "support": len(y_true),
        },
        {
            "target": target,
            "model": model,
            "fold": fold,
            "metric": "precision_macro",
            "value": precision,
            "label": "all",
            "support": len(y_true),
        },
        {
            "target": target,
            "model": model,
            "fold": fold,
            "metric": "recall_macro",
            "value": recall,
            "label": "all",
            "support": len(y_true),
        },
        {
            "target": target,
            "model": model,
            "fold": fold,
            "metric": "f1_macro",
            "value": f1,
            "label": "all",
            "support": len(y_true),
        },
    ]
    labels = sorted(pd.unique(y_true))
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    for label, p, r, score, n in zip(labels, class_precision, class_recall, class_f1, class_support):
        rows.extend(
            [
                {"target": target, "model": model, "fold": fold, "metric": "precision", "value": p, "label": label, "support": n},
                {"target": target, "model": model, "fold": fold, "metric": "recall", "value": r, "label": label, "support": n},
                {"target": target, "model": model, "fold": fold, "metric": "f1", "value": score, "label": label, "support": n},
            ]
        )
    return rows


def get_embedding_matrix(texts: pd.Series, model_name: str, batch_size: int) -> np.ndarray | None:
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model.encode(
            texts.tolist(),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    except ImportError:
        print("[warn] sentence-transformers is not installed; embedding models will be skipped.", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"[warn] could not load sentence embedding model '{model_name}': {exc}", file=sys.stderr)
        return None


def benchmark_target(
    data: pd.DataFrame,
    target: str,
    folds: int,
    seed: int,
    embeddings: np.ndarray | None,
    include_tfidf: bool,
) -> tuple[pd.DataFrame, str]:
    y = data[target].astype(str)
    groups = data["group_id"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    factories = classifier_factories()
    rows: list[dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(data, y, groups), start=1):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        overlap = set(train["group_id"]).intersection(test["group_id"])
        if overlap:
            raise RuntimeError(f"Leakage detected in fold {fold}: {len(overlap)} posts occur in both partitions.")

        if include_tfidf:
            vectorizer = build_tfidf_vectorizer()
            x_train_tfidf = vectorizer.fit_transform(train["model_text"])
            x_test_tfidf = vectorizer.transform(test["model_text"])

            for model_name in ["majority_baseline", "tfidf_logistic_regression", "tfidf_linear_svm"]:
                classifier = factories[model_name]()
                if model_name == "majority_baseline":
                    classifier.fit(np.zeros((len(train), 1)), train[target])
                    predicted = classifier.predict(np.zeros((len(test), 1)))
                else:
                    classifier.fit(x_train_tfidf, train[target])
                    predicted = classifier.predict(x_test_tfidf)
                rows.extend(metric_rows(test[target], predicted, target, model_name, fold))

        if embeddings is None:
            raise RuntimeError("Sentence embeddings are required unless --include-tfidf is used.")
        x_train_embeddings = embeddings[train_idx]
        x_test_embeddings = embeddings[test_idx]
        metadata_encoder, x_train_metadata = fit_structured_encoder(train)
        x_test_metadata = metadata_encoder.transform(test[STRUCTURED_COLUMNS])
        x_train_embedding_sparse = combine_embedding_and_structure(
            x_train_embeddings, x_train_metadata, dense=False
        )
        x_test_embedding_sparse = combine_embedding_and_structure(
            x_test_embeddings, x_test_metadata, dense=False
        )
        x_train_embedding_dense = combine_embedding_and_structure(
            x_train_embeddings, x_train_metadata, dense=True
        )
        x_test_embedding_dense = combine_embedding_and_structure(
            x_test_embeddings, x_test_metadata, dense=True
        )
        for model_name in ["embedding_logistic_regression", "embedding_linear_svm", "embedding_mlp"]:
            classifier = factories[model_name]()
            if model_name == "embedding_mlp":
                classifier.fit(x_train_embedding_dense, train[target])
                predicted = classifier.predict(x_test_embedding_dense)
            else:
                classifier.fit(x_train_embedding_sparse, train[target])
                predicted = classifier.predict(x_test_embedding_sparse)
            rows.extend(metric_rows(test[target], predicted, target, model_name, fold))

    metric_frame = pd.DataFrame(rows)
    summary = metric_frame.loc[metric_frame["label"].eq("all")].groupby(
        ["target", "model", "metric"], as_index=False
    ).agg(mean=("value", "mean"), std=("value", "std"))
    best = (
        summary.loc[summary["metric"].eq("f1_macro")]
        .sort_values("mean", ascending=False)
        .iloc[0]["model"]
    )
    return metric_frame, str(best)


def fit_production_model(data: pd.DataFrame, target: str, model_name: str, embeddings: np.ndarray | None, embedding_model: str) -> dict[str, object]:
    classifier = classifier_factories()[model_name]()
    if model_name == "majority_baseline":
        classifier.fit(np.zeros((len(data), 1)), data[target])
        return {"target": target, "model": model_name, "feature_type": "constant", "classifier": classifier}

    if model_name.startswith("tfidf"):
        vectorizer = build_tfidf_vectorizer()
        features = vectorizer.fit_transform(data["model_text"])
        classifier.fit(features, data[target])
        return {"target": target, "model": model_name, "feature_type": "tfidf", "vectorizer": vectorizer, "classifier": classifier}

    if model_name.startswith("embedding") and embeddings is not None:
        metadata_encoder, metadata_features = fit_structured_encoder(data)
        features = combine_embedding_and_structure(
            embeddings, metadata_features, dense=model_name == "embedding_mlp"
        )
        classifier.fit(features, data[target])
        return {
            "target": target,
            "model": model_name,
            "feature_type": "embedding_structured",
            "embedding_model": embedding_model,
            "metadata_encoder": metadata_encoder,
            "structured_columns": STRUCTURED_COLUMNS,
            "classifier": classifier,
        }

    raise RuntimeError(f"Cannot fit selected model '{model_name}'.")


def transform_for_artifact(data: pd.DataFrame, artifact: dict[str, object], embedding_cache: dict[str, object]) -> object:
    if artifact["feature_type"] == "constant":
        return np.zeros((len(data), 1))
    if artifact["feature_type"] == "tfidf":
        return artifact["vectorizer"].transform(data["model_text"])

    model_name = str(artifact["embedding_model"])
    if model_name not in embedding_cache:
        from sentence_transformers import SentenceTransformer

        embedding_cache[model_name] = SentenceTransformer(model_name)
    embeddings = embedding_cache[model_name].encode(
        data["text"].tolist(),
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    structured_features = artifact["metadata_encoder"].transform(
        data[artifact["structured_columns"]]
    )
    return combine_embedding_and_structure(
        embeddings,
        structured_features,
        dense=artifact["model"] == "embedding_mlp",
    )


def iter_unlabeled_applications(parquet_path: str, labels_path: str, batch_size: int):
    query = """
    WITH training_sample AS (
        SELECT merge_key, key_occurrence, frame_domain
        FROM read_csv_auto(?)
    ),
    frame_applications AS (
        SELECT
            p.merge_key,
            p.key_occurrence,
            p.source,
            p.category,
            p.post_date,
            p.post_year,
            p.post_month,
            p.text,
            p.focal_actor,
            frames.frame_domain,
            frames.frame_subframe,
            CASE WHEN p.category LIKE 'China_%' THEN 'China' WHEN p.category LIKE 'RF_%' THEN 'Russia' END AS source_country,
            CASE WHEN p.category LIKE '%Media%' THEN 'Media' WHEN p.category LIKE '%Diplomat%' THEN 'Diplomatic' END AS account_type
        FROM read_parquet(?) AS p
        CROSS JOIN LATERAL (
            VALUES ('AUT', p.aut_frame), ('DEM', p.dem_frame), ('WEST', p.west_frame)
        ) AS frames(frame_domain, frame_subframe)
        WHERE p.post_year BETWEEN 2020 AND 2025
          AND p.text IS NOT NULL AND trim(p.text) <> ''
          AND frames.frame_subframe NOT IN ('No Category', 'Invalid label')
    )
    SELECT a.*
    FROM frame_applications AS a
    ANTI JOIN training_sample AS t
        ON a.merge_key = t.merge_key
        AND a.key_occurrence = t.key_occurrence
        AND a.frame_domain = t.frame_domain
    WHERE a.source_country IS NOT NULL AND a.account_type IS NOT NULL
    """
    con = duckdb.connect()
    try:
        reader = con.execute(query, [labels_path, parquet_path]).fetch_record_batch(rows_per_batch=batch_size)
        for batch in reader:
            yield batch.to_pandas()
    finally:
        con.close()


def predict_rest(
    labels_path: str,
    parquet_path: str,
    output_path: str,
    batch_size: int,
    focal_artifact: dict[str, object],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"Prediction output already exists: {output}. Choose a new --prediction-out path.")

    writer = None
    embedding_cache: dict[str, object] = {}
    total = 0
    try:
        for raw_batch in iter_unlabeled_applications(parquet_path, labels_path, batch_size):
            batch = prepare_data(raw_batch)
            focal_mask = batch["focal_actor"].ne("None")
            batch["predicted_focal_valence"] = "Not Applicable"
            if focal_mask.any():
                focal_data = batch.loc[focal_mask]
                batch.loc[focal_mask, "predicted_focal_valence"] = focal_artifact["classifier"].predict(
                    transform_for_artifact(focal_data, focal_artifact, embedding_cache)
                )
            result = batch[[
                "merge_key", "key_occurrence", "source", "category", "post_date", "post_year", "post_month",
                "source_country", "account_type", "frame_domain", "frame_subframe", "focal_actor",
                "predicted_focal_valence",
            ]]
            table = pa.Table.from_pandas(result, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
            total += len(result)
            print(f"Predicted {total:,} post-frame applications", flush=True)
    finally:
        if writer is not None:
            writer.close()

    print(f"Wrote {total:,} predictions to {output}")


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise SystemExit(f"Labeled CSV not found: {labels_path}")

    if args.smoke_test and args.predict_rest:
        raise SystemExit("--smoke-test cannot be combined with --predict-rest")
    if args.smoke_test and args.out_dir == DEFAULT_OUT_DIR:
        args.out_dir = "derived/valence/models_smoke_test"
    if args.smoke_test:
        args.folds = 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_data = prepare_data(pd.read_csv(labels_path, low_memory=False))
    if args.smoke_test:
        full_data = (
            full_data.groupby(["source_country", "account_type", "frame_domain"], group_keys=False)
            .sample(n=50, random_state=args.seed)
            .reset_index(drop=True)
        )
        print(
            f"Smoke test: {len(full_data):,} rows from 12 strata; "
            f"grouped {args.folds}-fold cross-validation; output={out_dir}"
        )

    print(f"Encoding sentence embeddings once for {len(full_data):,} rows using {args.embedding_model}.")
    full_embeddings = get_embedding_matrix(
        full_data["text"], args.embedding_model, args.embedding_batch_size
    )
    if full_embeddings is None:
        raise SystemExit("Could not create sentence embeddings; verify --embedding-model points to a local model.")

    metrics_frames = []
    selected_models: dict[str, str] = {}
    target_data: dict[str, pd.DataFrame] = {}
    embedding_matrices: dict[str, np.ndarray | None] = {}

    for target, config in TARGETS.items():
        eligible_mask = ~full_data[target].isin(config["excluded_labels"])
        data = full_data.loc[eligible_mask].reset_index(drop=True)
        if data[target].nunique() < 2:
            raise SystemExit(f"Target '{target}' has fewer than two classes after exclusions.")
        target_data[target] = data
        print(f"\nTarget: {target}; rows={len(data):,}; classes={data[target].value_counts().to_dict()}")

        embeddings = full_embeddings[eligible_mask.to_numpy()]
        embedding_matrices[target] = embeddings
        metric_frame, best = benchmark_target(
            data,
            target,
            args.folds,
            args.seed,
            embeddings,
            args.include_tfidf,
        )
        metrics_frames.append(metric_frame)
        selected_models[target] = best
        print(f"Selected by mean grouped-CV macro F1: {target} -> {best}")

    metrics = pd.concat(metrics_frames, ignore_index=True)
    summary = metrics.loc[metrics["label"].eq("all")].groupby(
        ["target", "model", "metric"], as_index=False
    ).agg(mean=("value", "mean"), std=("value", "std"))
    metrics.to_csv(out_dir / "valence_cv_metrics_by_fold.csv", index=False)
    summary.to_csv(out_dir / "valence_cv_metrics_summary.csv", index=False)
    (out_dir / "selected_models.json").write_text(json.dumps(selected_models, indent=2) + "\n", encoding="utf-8")

    artifacts: dict[str, dict[str, object]] = {}
    for target, model_name in selected_models.items():
        artifact = fit_production_model(
            target_data[target],
            target,
            model_name,
            embedding_matrices[target],
            args.embedding_model,
        )
        artifact_path = out_dir / f"{target}_production_model.joblib"
        joblib.dump(artifact, artifact_path)
        artifacts[target] = artifact
        print(f"Saved {target} production model: {artifact_path}")

    print("\nGrouped cross-validation summary")
    print(
        summary.loc[summary["metric"].isin(["accuracy", "precision_macro", "recall_macro", "f1_macro"])]
        .pivot_table(index=["target", "model"], columns="metric", values="mean")
        .round(4)
        .to_string()
    )

    if args.predict_rest:
        predict_rest(
            labels_path=str(labels_path),
            parquet_path=args.parquet,
            output_path=args.prediction_out,
            batch_size=args.batch_size,
            focal_artifact=artifacts["focal_valence"],
        )


if __name__ == "__main__":
    main()
