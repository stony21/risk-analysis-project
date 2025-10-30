# excel_to_sqlite.py
import pandas as pd
import sqlite3
from pathlib import Path

# Поменяйте путь к файлу, если нужно
xlsx_path = Path("md_corporate_network_contagion.xlsx")
db_path = Path("corporate_network.db")

# Открываем Excel
xls = pd.ExcelFile(xlsx_path)

print("naidennie listi:", xls.sheet_names)

# Соединение с SQLite (файл создастся автоматически если не существует)
conn = sqlite3.connect(db_path)

# Для каждого листа — читаем и записываем в SQLite
for sheet in xls.sheet_names:
    df = xls.parse(sheet)
    # Опционально: приведение имён столбцов к удобному виду:
    # df.columns = [c.strip() for c in df.columns]
    table_name = sheet.replace(" ", "_")  # имя таблицы в БД
    print(f"zapisavaiu lista '{sheet}' -> tablita '{table_name}', strok: {len(df)}")
    df.to_sql(table_name, conn, if_exists="replace", index=False)

conn.close()
print("gotovo. BD:", db_path.resolve())
