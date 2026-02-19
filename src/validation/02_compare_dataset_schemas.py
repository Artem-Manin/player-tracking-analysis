import pandas as pd


def normalize_columns(cols):
    """
    Normalization rules:
    - lowercase
    - strip spaces
    - remove underscores
    - map known synonyms
    """

    synonym_map = {
        "cnt": "count",
    }

    normalized = set()

    for c in cols:
        c_norm = c.strip().lower()
        c_norm = c_norm.replace("_", "")

        if c_norm in synonym_map:
            c_norm = synonym_map[c_norm]

        normalized.add(c_norm)

    return normalized


def compare_schemas(file_a, file_b):

    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    cols_a = normalize_columns(df_a.columns)
    cols_b = normalize_columns(df_b.columns)

    print("\n=== DATASET A COLUMNS (normalized) ===")
    print(sorted(cols_a))

    print("\n=== DATASET B COLUMNS (normalized) ===")
    print(sorted(cols_b))

    print("\n=== COMMON COLUMNS ===")
    print(sorted(cols_a & cols_b))

    print("\n=== ONLY IN DATASET A ===")
    print(sorted(cols_a - cols_b))

    print("\n=== ONLY IN DATASET B ===")
    print(sorted(cols_b - cols_a))


if __name__ == "__main__":

    FILE_A = (
        "data/raw/"
        "new_player_data_2026_02_06_174048 - "
        "new_player_data_2026_02_06_174048.csv"
    )

    FILE_B = (
        "data/raw/"
        "player-activity-77686439338177232025102910010003-2026-02-16T20-43-45-776Z - "
        "player-activity-77686439338177232025102910010003-2026-02-16T20-43-45-776Z.csv"
    )

    compare_schemas(FILE_A, FILE_B)
