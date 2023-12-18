def dataset_helper(dataset: dict) -> dict:
    return {
        "id": str(dataset["_id"]),
        "name": dataset["name"],
        "description": dataset["description"],
        "task": dataset["task"],
        "items": dataset["items"],
    }

def item_helper(item: dict) -> dict:
    return {
        "id": str(item["_id"]),
        "dataset": item["dataset"],
        "file": item["file"],
        "annotations": item["annotations"],
        "entities": item["entities"],
        "isInTest": item["isInTest"],
        "relations": item["relations"],

    }
