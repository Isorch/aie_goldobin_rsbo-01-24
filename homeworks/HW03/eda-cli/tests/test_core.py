from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)

def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def _test_df_for_quality_heuristics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1001, 1002, 1003, 1003, 1005],  # Дубликат ID
            "country": ["RU", "RU", "RU", "RU", "RU"],  # Константная колонка
            "category": ["A", "A", "A", "A", "A"],  # Тоже константная
            "revenue": [0, 0, 0, 0, 0],  # Все значения нулевые
            "name": [f"User_{i}" for i in range(5)],  # Все уникальные значения
            "value": [1.5, 2.3, 3.1, None, 4.2],  # Нормальная числовая колонка
        }
    )



def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

# Для новых эвристик
def test_quality_flags_new_heuristics():
    df = _test_df_for_quality_heuristics()
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    # 1 Проверка константных колонок
    assert flags["has_constant_columns"] is True
    assert "country" in flags["constant_columns"]
    assert "category" in flags["constant_columns"]

    # 2 Проверка дубликатов ID
    assert flags["has_suspicious_id_duplicates"] is True
    assert len(flags["id_duplicates_info"]) > 0
    id_found = False
    for info in flags["id_duplicates_info"]:
        if info["name"] == "user_id":
            assert info["duplicate_rate"] > 0
            id_found = True
            break
    assert id_found

    # 3 Проверка высокой кардинальности категориальных признаков
    assert flags["has_high_cardinality_categoricals"] is True
    assert len(flags["high_cardinality_columns"]) > 0
    high_card_found = False
    for col in flags["high_cardinality_columns"]:
        if col["name"] == "name":
            assert col["unique"] == 5
            high_card_found = True
            break
    assert high_card_found
    
    # 4 Проверка множества нулевых значений
    assert flags["has_many_zero_values"] is True
    assert len(flags["many_zero_columns"]) > 0
    zero_found = False
    for col in flags["many_zero_columns"]:
        if col["name"] == "revenue":
            assert col["zero_share"] == 1.0  # min=0 и max=0 для всех значений
            zero_found = True
            break
    assert zero_found

    # 5 Проверка общего скора качества (должен быть понижен из-за проблем)
    assert 0.0 <= flags["quality_score"] <= 1.0
    assert flags["quality_score"] < 1.0


def test_quality_flags_no_issues():
    # Создаем DataFrame с большим количеством строк, чтобы избежать ложного срабатывания высокой кардинальности (нужно больше 100 строк, чтобы порог был выше 5)
    df = pd.DataFrame({
        "id": list(range(1, 101)),  # 100 уникальных ID
        "value": list(range(10, 1010, 10)),  # 100 уникальных значений
        "category": ["A", "B", "C", "D", "E"] * 20,  # 5 уникальных значений из 100 строк - 5%
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Все флаги должны быть False для чистых данных
    assert flags["has_constant_columns"] is False
    assert flags["has_suspicious_id_duplicates"] is False
    assert flags["has_high_cardinality_categoricals"] is False
    assert flags["has_many_zero_values"] is False
    
    assert flags["quality_score"] > 0.7


def test_quality_flags_mixed_scenario():
    # Увеличим количество строк до 10, чтобы избежать срабатывания высокой кардинальности при нормальной категориальной колонке
    df = pd.DataFrame({
        "user_id": list(range(1, 11)),  # 10 уникальных - OK
        "status": ["active"] * 10,  # Константная колонка - проблема
        "score": list(range(85, 95)),  # Нормальная числовая
        # Категориальная колонка с 4 уникальными значениями из 10 - 40%
        "category": ["A", "A", "B", "B", "C", "C", "D", "D", "A", "B"],
        "zero_col": [0, 0, 1, 2, 3, 0, 1, 2, 0, 3],  # Не все нули - OK
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    assert flags["has_constant_columns"] is True
    assert flags["has_suspicious_id_duplicates"] is False
    assert flags["has_high_cardinality_categoricals"] is False
    assert flags["has_many_zero_values"] is False
    
    # Проверяем, что только статус отмечен как константная колонка
    assert "status" in flags["constant_columns"]
    assert len(flags["constant_columns"]) == 1
