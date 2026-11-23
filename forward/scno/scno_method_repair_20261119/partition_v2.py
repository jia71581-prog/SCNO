"""Apply a sealed, train-only-normalized 900/100 partition to the centered release."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from data_v2 import _file_sha, _json_sha


SCHEMA = "SCNO_UL_CENTERED_900_100_PARTITION_V1"


def apply_partition(corpus, manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or corpus.kind != "centered_uniform_layered":
        raise ValueError("the centered 900/100 partition requires its matching SCNO source release")
    source = manifest["source_release"]
    if str(corpus.root) != source["root"]:
        raise ValueError("partition points to another published source release")
    for key, filename in (("marker_sha256", "DATASET_COMPLETE.json"),
                          ("records_sha256", "RECORDS.json"),
                          ("published_normalization_sha256", "normalization_ul.json")):
        if _file_sha(corpus.root / filename) != source[key]:
            raise ValueError("the source release differs from the sealed partition")
    if _json_sha(list(corpus.train_ids)) != source["original_train_ids_sha256"]:
        raise ValueError("open the complete source release before applying its partition")
    if tuple(corpus.times.shape) != (401,) or not np.array_equal(corpus.times, np.arange(401) * .0025):
        raise ValueError("the partition requires all native 401 pressure frames from 0 to 1 second")
    train, test = (tuple(map(str, manifest[name])) for name in ("train_ids", "test_ids"))
    if len(train) != 900 or len(test) != 100 or len(set(train + test)) != 1000:
        raise ValueError("the partition must contain exactly 900 distinct TRAIN and 100 distinct TEST records")
    if set(train + test) != set(corpus.train_ids):
        raise ValueError("partition membership differs from the original 1,000 records")
    for role, ids, expected in (("train", train, 450), ("test", test, 50)):
        for family in ("uniform", "layered"):
            if sum(corpus.records[k]["family"] == family for k in ids) != expected:
                raise ValueError("both roles require the declared balanced medium families")
        for key in ids:
            if not np.array_equal(corpus._bound[key].source[:2], [1000., 1000.]):
                raise ValueError("every partitioned record must use the declared central source")
    normalizer_path = manifest_path.parent / manifest["normalization_train_only"]["file"]
    if _file_sha(normalizer_path) != manifest["normalization_train_only"]["sha256"]:
        raise ValueError("the TRAIN900 normalizer checksum changed")
    normalizer = json.loads(normalizer_path.read_text(encoding="utf-8"))
    if (normalizer["record_count"] != 900 or normalizer["train_manifest_sha256"] != _json_sha(list(train))
            or normalizer["validation_test_labels_used"] is not False):
        raise ValueError("the normalizer must be fitted exclusively to this TRAIN900 roster")
    original_binding = corpus.binding
    records = {}
    bound = {}
    for role, ids in (("train", train), ("test", test)):
        for key in ids:
            information = {**corpus.records[key], "split": role,
                           "original_release_split": corpus.records[key]["split"]}
            records[key] = information
            bound[key] = replace(corpus._bound[key], information=information)
    corpus.records, corpus._bound = records, bound
    corpus.train_ids, corpus.validation_ids, corpus.test_ids = train, (), test
    corpus.metadata = {**corpus.metadata, **normalizer,
                       "velocity_mean_mps": normalizer["velocity_center_mps"],
                       "velocity_std_mps": normalizer["velocity_scale_mps"]}
    corpus._check_metadata()
    corpus.binding = {
        **original_binding,
        "selected_ids": {"train": list(train), "validation": [], "test": list(test)},
        "normalization_sha256": _json_sha(corpus.metadata),
        "partition": {"schema": SCHEMA, "manifest": str(manifest_path),
                      "sha256": _file_sha(manifest_path),
                      "original_binding_sha256": _json_sha(original_binding),
                      "role_definition": "900 TRAIN and 100 held-out TEST, central source"},
    }
    corpus.partition_manifest = manifest
    return corpus
