from chroma_helpers import COLLECTION_NAME, build_db_from_docs, load_db


def reload_data() -> None:
    db = load_db()
    db.delete_collection()
    print(f"Deleted existing Chroma collection: {COLLECTION_NAME}")

    build_db_from_docs()
    print("Reload complete.")


if __name__ == "__main__":
    reload_data()
