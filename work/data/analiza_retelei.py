import sqlite3
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# === 1. Подключаемся к базе данных ===
db_path = "D:/work/data/corporate_network.db"
conn = sqlite3.connect(db_path)

# === 2. Загружаем таблицы из SQLite ===
companii = pd.read_sql_query("SELECT * FROM Companii", conn)
relatii = pd.read_sql_query("SELECT * FROM Relatii", conn)

# === 3. Сохраняем их в CSV (на случай анализа в Excel) ===
companii.to_csv("D:/work/data/Companii.csv", index=False)
relatii.to_csv("D:/work/data/Relatii.csv", index=False)

print("✅ Таблицы экспортированы в CSV!")

# === 4. Создаём направленный граф ===
G = nx.DiGraph()

# Добавляем компании как узлы
for _, row in companii.iterrows():
    G.add_node(row["company_id"],
               name=row.get("company_name", ""),
               sector=row.get("sector", ""),
               region=row.get("region", ""),
               size=row.get("size_category", ""))

# Добавляем связи как рёбра
for _, row in relatii.iterrows():
    G.add_edge(row["source_company_id"],
               row["target_company_id"],
               relation=row.get("relation_type", ""),
               weight=row.get("exposure_mdl", 0.0),
               ownership=row.get("ownership_pct", 0.0))

print(f"📊 Всего компаний: {G.number_of_nodes()}, связей: {G.number_of_edges()}")

# === 5. Анализ: самые "влиятельные" компании (по исходящим связям) ===
out_degree = sorted(G.out_degree(weight="weight"), key=lambda x: x[1], reverse=True)[:10]
print("\n🏢 Топ-10 компаний по количеству исходящих связей:")
for comp, deg in out_degree:
    name = companii.loc[companii["company_id"] == comp, "company_name"].values
    print(f"{comp} - {name[0] if len(name)>0 else 'Неизвестно'} ({deg:.0f} связей)")

# === 6. Анализ: самые "зависимые" компании (по входящим связям) ===
in_degree = sorted(G.in_degree(weight="weight"), key=lambda x: x[1], reverse=True)[:10]
print("\n🏦 Топ-10 компаний по количеству входящих связей:")
for comp, deg in in_degree:
    name = companii.loc[companii["company_id"] == comp, "company_name"].values
    print(f"{comp} - {name[0] if len(name)>0 else 'Неизвестно'} ({deg:.0f} связей)")

# === 7. Визуализация ===
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=0.25, iterations=40)
nx.draw_networkx_nodes(G, pos, node_size=30, node_color="skyblue", alpha=0.7)
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=8, edge_color="gray", alpha=0.4)
plt.title("🌐 Сеть взаимосвязей компаний", fontsize=16)
plt.axis("off")
plt.show()

# === 8. Дополнительно: центральность (влияние узла) ===
centrality = nx.degree_centrality(G)
top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("\n💡 Самые центральные компании (наиболее влиятельные):")
for comp, cent in top_central:
    name = companii.loc[companii["company_id"] == comp, "company_name"].values
    print(f"{comp} - {name[0] if len(name)>0 else 'Неизвестно'} (центральность: {cent:.3f})")
