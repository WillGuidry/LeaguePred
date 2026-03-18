# Oracle's Elixir Data Sources

Download links for Oracle's Elixir match data CSVs (Google Drive).

## Direct Download Links

| Year | Google Drive Link | Direct Download |
|------|------------------|-----------------|
| 2025 (current) | https://drive.google.com/file/d/1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm/view | `curl -L "https://drive.google.com/uc?id=1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm&export=download&confirm=t" -o oracle_elixir_2025.csv` |
| 2025 (alt) | https://drive.google.com/file/d/1v6LRphp2kYciU4SXp0PCjEMuev1bDejc/view | `curl -L "https://drive.google.com/uc?id=1v6LRphp2kYciU4SXp0PCjEMuev1bDejc&export=download&confirm=t" -o oracle_elixir_2025_alt.csv` |
| 2024 | https://drive.google.com/file/d/1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN/view | `curl -L "https://drive.google.com/uc?id=1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN&export=download&confirm=t" -o oracle_elixir_2024.csv` |
| 2023 | https://drive.google.com/file/d/1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2/view | `curl -L "https://drive.google.com/uc?id=1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2&export=download&confirm=t" -o oracle_elixir_2023.csv` |
| 2022 | https://drive.google.com/file/d/1EHmptHyzY8owv0BAcNKtkQpMwfkURwRy/view | `curl -L "https://drive.google.com/uc?id=1EHmptHyzY8owv0BAcNKtkQpMwfkURwRy&export=download&confirm=t" -o oracle_elixir_2022.csv` |
| 2021 | https://drive.google.com/file/d/1fzwTTz77hcnYjOnO9ONeoPrkWCoOSecA/view | `curl -L "https://drive.google.com/uc?id=1fzwTTz77hcnYjOnO9ONeoPrkWCoOSecA&export=download&confirm=t" -o oracle_elixir_2021.csv` |

## File IDs (for programmatic access)

```python
ORACLE_ELIXIR_FILE_IDS = {
    "2025": "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm",
    "2025_alt": "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",
    "2024": "1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN",
    "2023": "1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2",
    "2022": "1EHmptHyzY8owv0BAcNKtkQpMwfkURwRy",
    "2021": "1fzwTTz77hcnYjOnO9ONeoPrkWCoOSecA",
}
```

## Notes

- Data format: 12 rows per game (10 players + 2 team summary rows)
- Position column: "top", "jng", "mid", "bot", "sup", "team"
- Files from 2020+ are CSV; older files are XLSX
- Source: [Oracle's Elixir](https://oracleselixir.com/tools/downloads)
