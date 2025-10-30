# app.py — Ego-Graph корпоративной сети (Streamlit + PyVis)
# Запуск: streamlit run app.py

import os
import math
import sqlite3
import tempfile
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit as st

# =========================
#  Настройки / путь к БД
# =========================
DB_PATH = r"C:\Users\maksi\Desktop\datas\corporate_network.db"   # ← при желании измени

# =========================
#  Вспомогательные функции
# =========================
def open_conn():
    return sqlite3.connect(DB_PATH)

def list_tables(conn) -> list:
    q = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    try:
        return pd.read_sql_query(q, conn)["name"].tolist()
    except Exception:
        return []

def _pick_table(actual_names_lower: list, preferred_candidates: list) -> str | None:
    # 1) точное совпадение по lower
    for cand in preferred_candidates:
        if cand.lower() in actual_names_lower:
            idx = actual_names_lower.index(cand.lower())
            return actual_names_lower_src[idx]
    # 2) частичное совпадение
    for i, name in enumerate(actual_names_lower):
        for cand in preferred_candidates:
            if cand.lower() in name:
                return actual_names_src[i]
    return None

def detect_tables(conn):
    """Автоопределение имён таблиц компаний и связей."""
    names = list_tables(conn)
    lower = [n.lower() for n in names]
    # сохраняем исходные списки для восстановления оригинального регистра
    global actual_names_lower, actual_names_lower_src, actual_names_src
    actual_names_lower = lower
    actual_names_lower_src = names[:]
    actual_names_src = names[:]

    comp_candidates = ["Companii", "Companies", "Company", "Companie", "firms", "nodes"]
    rel_candidates  = ["Relatii", "Relations", "Edges", "Links", "edges"]

    tbl_comp = _pick_table(lower, comp_candidates)
    tbl_rel  = _pick_table(lower, rel_candidates)
    return tbl_comp, tbl_rel

# -------------------------
# Диагностика подключения
# -------------------------
st.set_page_config(page_title="Ego Graph Viewer", layout="wide")
st.title("Ego-Graph корпоративной сети")

# --- Session state init ---
if "center_id" not in st.session_state:
    st.session_state.center_id = None
if "nodes_df" not in st.session_state:
    st.session_state.nodes_df = None
if "edges_df" not in st.session_state:
    st.session_state.edges_df = None
if "infected_order" not in st.session_state:
    st.session_state.infected_order = None
if "loss_table" not in st.session_state:
    st.session_state.loss_table = None
# расширенный подграф для каскада
if "cascade_nodes_df" not in st.session_state:
    st.session_state.cascade_nodes_df = None
if "cascade_edges_df" not in st.session_state:
    st.session_state.cascade_edges_df = None

try:
    conn_diag = open_conn()
    all_tables = list_tables(conn_diag)
    tbl_comp, tbl_rel = detect_tables(conn_diag)
    info_line = f"Подключено к БД: {DB_PATH} · таблиц: {len(all_tables)}"

    if tbl_comp:
        try:
            n_companies = pd.read_sql_query(f"SELECT COUNT(*) n FROM {tbl_comp};", conn_diag)["n"].iloc[0]
        except Exception:
            n_companies = "?"
        info_line += f" · компаний (из {tbl_comp}): {n_companies}"
    if tbl_rel:
        info_line += f" · связи: {tbl_rel}"

    st.success(info_line)
    conn_diag.close()

    if not (tbl_comp and tbl_rel):
        st.error(
            "Файл открыт, но не нашёл подходящие таблицы компаний/связей.\n\n"
            f"Есть таблицы: {all_tables}\n\n"
            "Ожидались названия вроде Companii/Relatii (или их аналоги). "
            "Либо укажи правильный DB_PATH, либо импортируй CSV в SQLite."
        )
        st.stop()
except Exception as e:
    st.error(f"Ошибка подключения к БД: {e}")
    st.stop()

# =========================
#  Поиск компаний
# =========================
@st.cache_data(show_spinner=False)
def search_companies(q: str) -> pd.DataFrame:
    q = (q or "").strip()
    if not q:
        return pd.DataFrame(columns=["company_id", "company_name"])
    conn = open_conn()
    try:
        sql = f"""
            SELECT company_id, company_name
            FROM {tbl_comp}
            WHERE lower(company_name) LIKE lower(?)
               OR lower(company_id) = lower(?)
            ORDER BY company_name
            LIMIT 50;
        """
        return pd.read_sql_query(sql, conn, params=[f"%{q}%", q.lower()])
    finally:
        conn.close()

def get_ego(center_id: str, direction: str, min_exposure: float,
            type_filters: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает (nodes_df, edges_df) для эго-сети.
    direction: 'both' | 'out' | 'in'
    type_filters: список типов (loan, trade_credit, partnership, ownership)
    """
    conn = open_conn()
    try:
        where_dir = []
        params = []

        if direction == "out":
            where_dir.append("r.source_company_id = ?")
            params.append(center_id)
        elif direction == "in":
            where_dir.append("r.target_company_id = ?")
            params.append(center_id)
        else:  # both
            where_dir.append("(r.source_company_id = ? OR r.target_company_id = ?)")
            params.extend([center_id, center_id])

        where_parts = [" AND ".join(where_dir)] if where_dir else []

        if min_exposure and min_exposure > 0:
            where_parts.append("r.exposure_mdl >= ?")
            params.append(float(min_exposure))

        if type_filters:
            qmarks = ",".join(["?"] * len(type_filters))
            where_parts.append(f"lower(r.relation_type) IN ({qmarks})")
            params.extend([t.lower() for t in type_filters])

        where_sql = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        sql_edges = f"""
            SELECT
                r.source_company_id AS source,
                r.target_company_id AS target,
                r.exposure_mdl      AS weight,
                r.relation_type     AS rel_type,
                r.lgd_assumption    AS lgd
            FROM {tbl_rel} r
            {where_sql};
        """
        edges = pd.read_sql_query(sql_edges, conn, params=params)

        ids = set([center_id]) | set(edges["source"].dropna()) | set(edges["target"].dropna())
        if not ids:
            return pd.DataFrame(columns=["company_id","company_name","equity_mdl"]), edges

        qmarks = ",".join(["?"] * len(ids))
        sql_nodes = f"""
            SELECT company_id, company_name, equity_mdl
            FROM {tbl_comp}
            WHERE company_id IN ({qmarks});
        """
        nodes = pd.read_sql_query(sql_nodes, conn, params=list(ids))
        return nodes, edges
    finally:
        conn.close()

def get_out_subgraph_khops(center_id: str, max_hops: int, min_exposure: float,
                           type_filters: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Собирает ориентированный подграф исходящих связей до K-хопов от center_id.
    Учитывает min_exposure и type_filters.
    Возвращает (nodes_df, edges_df).
    """
    conn = open_conn()
    try:
        all_ids = set([center_id])
        frontier = set([center_id])
        edges_all = []

        for _ in range(max(1, int(max_hops))):
            if not frontier:
                break
            qmarks = ",".join(["?"] * len(frontier))
            params = list(frontier)

            where_parts = [f"r.source_company_id IN ({qmarks})"]

            if min_exposure and min_exposure > 0:
                where_parts.append("r.exposure_mdl >= ?")
                params.append(float(min_exposure))

            if type_filters:
                q2 = ",".join(["?"] * len(type_filters))
                where_parts.append(f"lower(r.relation_type) IN ({q2})")
                params.extend([t.lower() for t in type_filters])

            sql = f"""
                SELECT
                    r.source_company_id AS source,
                    r.target_company_id AS target,
                    r.exposure_mdl      AS weight,
                    r.relation_type     AS rel_type,
                    r.lgd_assumption    AS lgd
                FROM {tbl_rel} r
                WHERE {" AND ".join(where_parts)};
            """
            chunk = pd.read_sql_query(sql, conn, params=params)
            if chunk.empty:
                break

            edges_all.append(chunk)
            next_frontier = set(chunk["target"].dropna().unique()) - all_ids
            all_ids |= set(chunk["source"].dropna().unique())
            all_ids |= set(chunk["target"].dropna().unique())
            frontier = next_frontier

        edges_df = pd.concat(edges_all, ignore_index=True).drop_duplicates() if edges_all else pd.DataFrame(
            columns=["source","target","weight","rel_type","lgd"]
        )

        if all_ids:
            qmarks = ",".join(["?"] * len(all_ids))
            sql_nodes = f"""
                SELECT company_id, company_name, equity_mdl
                FROM {tbl_comp}
                WHERE company_id IN ({qmarks});
            """
            nodes_df = pd.read_sql_query(sql_nodes, conn, params=list(all_ids))
        else:
            nodes_df = pd.DataFrame(columns=["company_id","company_name","equity_mdl"])

        return nodes_df, edges_df
    finally:
        conn.close()

# =========================
#  Граф и симуляция заражения
# =========================
def _build_nx_graph(center_id: str, nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    """Строим ориентированный граф с нужными атрибутами для расчётов."""
    G = nx.DiGraph()
    for _, r in nodes.iterrows():
        G.add_node(
            r["company_id"],
            label=r.get("company_name", r["company_id"]),
            equity=float(r.get("equity_mdl", 0.0) or 0.0),
        )
    for _, r in edges.iterrows():
        G.add_edge(
            r["source"], r["target"],
            weight=float(r.get("weight", 0.0) or 0.0),  # exposure_mdl
            rel_type=(r.get("rel_type") or ""),
            lgd=float(r.get("lgd", 0.0) or 0.0),        # LGD
        )
    return G

def simulate_contagion(G: nx.DiGraph, start_node: str, threshold: float = 0.25, max_hops: int = 3):
    """
    Каскад заражения с волнами.
    Возвращает:
      infected_order: dict[node] = номер волны (0 — стартовый)
      loss_table: DataFrame ['company_id','company_name','equity_mdl','abs_loss_mdl','rel_loss','wave']
    """
    infected_order = {start_node: 0}
    losses_acc = {}
    waves = {0: {start_node}}

    hop = 0
    while hop < max_hops and waves.get(hop):
        next_wave = set()
        for a in waves[hop]:
            for b in G.successors(a):
                if b in infected_order:
                    continue
                exposure = float(G[a][b].get("weight", 0.0) or 0.0)
                lgd_a    = float(G[a][b].get("lgd", 0.0) or 0.0)
                loss_b   = exposure * lgd_a

                equity_b = float(G.nodes[b].get("equity", 0.0) or 0.0)
                rel_loss = float("inf") if (equity_b <= 0 and loss_b > 0) else (loss_b / equity_b if equity_b > 0 else 0.0)

                losses_acc[b] = losses_acc.get(b, 0.0) + loss_b
                if rel_loss > threshold:
                    infected_order[b] = hop + 1
                    next_wave.add(b)
        hop += 1
        if next_wave:
            waves[hop] = next_wave

    rows = []
    for b, abs_loss in losses_acc.items():
        rows.append({
            "company_id": b,
            "company_name": G.nodes[b].get("label", b),
            "equity_mdl": G.nodes[b].get("equity", 0.0),
            "abs_loss_mdl": abs_loss,
            "rel_loss": (abs_loss / G.nodes[b]["equity"]) if (G.nodes[b].get("equity", 0.0) or 0.0) > 0 else float("inf"),
            "wave": infected_order.get(b, None),
        })
    loss_table = pd.DataFrame(rows).sort_values(["wave", "rel_loss", "abs_loss_mdl"], ascending=[True, False, False])
    return infected_order, loss_table

def build_html(center_id: str, nodes: pd.DataFrame, edges: pd.DataFrame,
               infected_order: dict | None = None,
               loss_table: pd.DataFrame | None = None) -> str:
    """Рендер интерактивного графа PyVis (подсветка волн заражения и тултипы потерь)."""
    G = _build_nx_graph(center_id, nodes, edges)

    loss_map_abs = {}
    loss_map_rel = {}
    if loss_table is not None and not loss_table.empty:
        loss_map_abs = dict(zip(loss_table["company_id"], loss_table["abs_loss_mdl"]))
        loss_map_rel = dict(zip(loss_table["company_id"], loss_table["rel_loss"]))

    net = Network(height="820px", width="100%", directed=True, bgcolor="#111", font_color="#eee")
    net.barnes_hut(gravity=-30000, central_gravity=0.3, spring_length=150, spring_strength=0.01)

        # палитра для волн заражения: 0 — центр, 1.. — волны
    wave_colors = {
        0: "#aa9800",  # центр (оранжевый)
        1: "#e53935",  # волна 1 — красный
        2: "#fb8c00",  # волна 2 — янтарный
        3: "#fdd835",  # волна 3 — жёлтый
        4: "#8e24aa",  # волна 4 — фиолетовый
        5: "#1e88e5",  # волна 5 — синий
    }
    default_node_color = "#61dafb"

    def color_by_risk(risk_pct: float) -> str:
        if risk_pct < 10:
            return "#2e7d32"  # green
        if risk_pct < 30:
            return "#1e88e5"  # blue
        if risk_pct < 60:
            return "#fb8c00"  # orange
        if risk_pct < 80:
            return "#e53935"  # red
        return "#9e9e9e"     # gray

    # узлы
    for n, d in G.nodes(data=True):
        # суммарный вес связей для размера
        total_w = 0.0
        for _, _, e in G.in_edges(n, data=True):
            total_w += float(e.get("weight", 0.0))
        for _, _, e in G.out_edges(n, data=True):
            total_w += float(e.get("weight", 0.0))

        size = 30 if n == center_id else 12 + (math.log10(total_w + 1.0) * 5.0)

        # риск = rel_loss * 100, если он есть в loss_table; центр всегда красный
        if n == center_id:
            color = "#e53935"  # красный для изначально падающей компании
            risk_pct = 100.0
        else:
            rel = loss_map_rel.get(n, None)  # rel_loss как доля
            if rel is None:
                color = default_node_color
                risk_pct = 0.0
            else:
                risk_pct = float(rel) * 100.0
                color = color_by_risk(risk_pct)

        equity = d.get("equity", 0.0)
        abs_loss = loss_map_abs.get(n, 0.0)
        risk_line = f"<br><span>Risk: {risk_pct:.1f}%</span>"
        loss_line = f"<br><span>Loss: {abs_loss:,.0f} MDL</span>" if abs_loss and abs_loss > 0 else ""

        title = f"<b>{d.get('label','')}</b><br>Equity: {equity:,.0f}{risk_line}{loss_line}"
        net.add_node(n, label=d.get("label",""), title=title, value=total_w, size=size, color=color)

    # рёбра
    edge_colors = {"loan": "#00e676", "trade_credit": "#ffd600", "ownership": "#bb86fc", "partnership": "#ff8a80"}
    for u, v, e in G.edges(data=True):
        rel = (e.get("rel_type") or "").lower()
        w   = float(e.get("weight", 0.0))
        lgd = float(e.get("lgd", 0.0))
        color = edge_colors.get(rel, "#9e9e9e")
        title = f"{rel or 'relation'}<br>exposure: {w:,.0f} MDL<br>LGD: {lgd:.2f}"
        net.add_edge(u, v, value=max(w, 1.0), color=color, title=title)

    net.set_options("""
    {
      "nodes": { "shape": "dot" },
      "edges": { "smooth": true, "arrows": { "to": { "enabled": true, "scaleFactor": 0.8 } } },
      "physics": { "stabilization": true }
    }
    """)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.write_html(tmp.name, open_browser=False, notebook=False)
    return tmp.name
# =========================
#  UI — поиск и выбор
# =========================
st.subheader("Поиск компании")
q = st.text_input("Введите часть названия ИЛИ точный company_id (например, CO102209):", "")

with st.expander("Фильтры графа", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        direction = st.radio("Направление связей", options=["both", "out", "in"], index=0, horizontal=True)
    with col2:
        min_exposure = st.number_input("Минимальная экспозиция (MDL)", min_value=0.0, value=0.0, step=1000.0)
    with col3:
        type_filters = st.multiselect(
            "Типы связей",
            options=["loan", "trade_credit", "partnership", "ownership"],
            default=[]
        )

if q:
    matches = search_companies(q)
    if matches.empty:
        st.warning("Ничего не найдено. Проверь орфографию или попробуй по company_id.")
    else:
        choice = st.selectbox(
            "Выберите компанию:",
            options=(matches["company_id"] + " — " + matches["company_name"]).tolist(),
            key="company_select"
        )
        if st.button("Показать эго-сеть", type="primary", key="btn_show"):
            center_id = choice.split(" — ")[0]
            nodes_df, edges_df = get_ego(center_id, direction, min_exposure, type_filters)
            if edges_df.empty:
                st.info("Для выбранных фильтров связей не найдено.")
            else:
                # сохраняем выбор
                st.session_state.center_id = center_id
                st.session_state.nodes_df = nodes_df
                st.session_state.edges_df = edges_df
                # сбрасываем прошлую симуляцию и расширенный подграф
                st.session_state.infected_order = None
                st.session_state.loss_table = None
                st.session_state.cascade_nodes_df = None
                st.session_state.cascade_edges_df = None
else:
    st.info("Начните с ввода части названия компании или точного company_id.")

# =========================
#  Постоянный рендер графа и симуляции
# =========================
if st.session_state.center_id and st.session_state.nodes_df is not None and st.session_state.edges_df is not None:
    # Текущая визуализация (если симуляция уже была — подсветка сохранится)
    html_file = build_html(
        st.session_state.center_id,
        st.session_state.nodes_df,
        st.session_state.edges_df,
        infected_order=st.session_state.infected_order,
        loss_table=st.session_state.loss_table
    )
    with open(html_file, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=840, scrolling=True)
    st.caption("Размер узла ~ суммарному весу связей; цвет ребра = тип связи; толщина = exposure_mdl.")

    st.divider()
    st.subheader("🧠 Симуляция потерь при дефолте выбранной компании и каскада заражения")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        threshold = st.slider("Порог относительной потери для дефолта (%)", min_value=5, max_value=80, value=25, step=5, key="th_slider")
    with colB:
        max_hops = st.number_input("Макс. число волн каскада", min_value=1, max_value=10, value=3, step=1, key="hops_num")
    with colC:
        run_sim = st.button("Смоделировать заражение", type="secondary", key="btn_sim")
    with colD:
        reset_sim = st.button("Сбросить симуляцию", key="btn_reset_sim")

    if reset_sim:
        st.session_state.infected_order = None
        st.session_state.loss_table = None
        st.session_state.cascade_nodes_df = None
        st.session_state.cascade_edges_df = None
        st.experimental_rerun()

    if run_sim:
        # строим каскадный подграф до max_hops исходящих шагов
        nodes_k, edges_k = get_out_subgraph_khops(
            st.session_state.center_id,
            max_hops=int(max_hops),
            min_exposure=min_exposure,
            type_filters=type_filters
        )
        # если пусто — используем эго-граф как fallback
        if edges_k is None or edges_k.empty:
            nodes_k, edges_k = st.session_state.nodes_df, st.session_state.edges_df

        st.session_state.cascade_nodes_df = nodes_k
        st.session_state.cascade_edges_df = edges_k

        # каскад на расширенном подграфе
        Gx = _build_nx_graph(st.session_state.center_id, nodes_k, edges_k)
        infected_order, loss_table = simulate_contagion(
            Gx,
            start_node=st.session_state.center_id,
            threshold=threshold / 100.0,
            max_hops=int(max_hops)
        )
        st.session_state.infected_order = infected_order
        st.session_state.loss_table = loss_table

        # если есть результаты — показываем
        # если есть результаты — показываем
    if st.session_state.loss_table is not None:
        loss_table = st.session_state.loss_table

        st.markdown("### 📊 Результаты симуляции заражения")

        # ✅ фильтр: показать только заражённые компании
        show_only_infected = st.toggle("Показать только компании, которые обанкротились (infected)", value=True)

        if not loss_table.empty:
            show = loss_table.copy()
            show["abs_loss_mdl"] = show["abs_loss_mdl"].round(0).astype(int)
            show["rel_loss_%"] = (show["rel_loss"] * 100).round(1)

            # если включен фильтр — оставляем только тех, у кого wave не None
            if show_only_infected:
                show = show[show["wave"].notnull() & (show["wave"] >= 1)]

            show = show.rename(columns={
                "company_id": "Компания ID",
                "company_name": "Компания",
                "equity_mdl": "Equity (MDL)",
                "abs_loss_mdl": "Потеря (MDL)",
                "rel_loss_%": "Потеря (%)",
                "wave": "Волна"
            })[["Компания ID","Компания","Equity (MDL)","Потеря (MDL)","Потеря (%)","Волна"]]

            st.dataframe(show, use_container_width=True, height=400)
        else:
            st.info("Потерь не зафиксировано для соседей при заданном пороге.")

        st.markdown("**Визуализация каскада:** стартовая компания = оранжевая; заражённые по волнам — оттенки красного/жёлтого.**")

        # --- безопасный выбор DataFrame для визуализации ---
        if st.session_state.cascade_nodes_df is not None and not st.session_state.cascade_nodes_df.empty:
            nodes_for_viz = st.session_state.cascade_nodes_df
        else:
            nodes_for_viz = st.session_state.nodes_df

        if st.session_state.cascade_edges_df is not None and not st.session_state.cascade_edges_df.empty:
            edges_for_viz = st.session_state.cascade_edges_df
        else:
            edges_for_viz = st.session_state.edges_df
        # ---------------------------------------------------

        # ✅ Если выбрано "показать только обанкротившихся" — фильтруем узлы и рёбра
        if show_only_infected and st.session_state.infected_order:
            infected_nodes = {n for n, wave in st.session_state.infected_order.items() if wave >= 1}
            infected_nodes.add(st.session_state.center_id)  # центр всегда показываем
            nodes_for_viz = nodes_for_viz[nodes_for_viz["company_id"].isin(infected_nodes)]
            edges_for_viz = edges_for_viz[
                edges_for_viz["source"].isin(infected_nodes) & edges_for_viz["target"].isin(infected_nodes)
            ]

        html_file2 = build_html(
            st.session_state.center_id,
            nodes_for_viz,
            edges_for_viz,
            infected_order=st.session_state.infected_order,
            loss_table=st.session_state.loss_table
        )

        with open(html_file2, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=840, scrolling=True)

# =========================
#  🧨 Системный риск — вкладка со сканированием БД
# =========================

st.divider()
tab_graph, tab_sysrisk = st.tabs(["📈 Граф/симуляция", "🧨 Системный риск (скан БД)"])

with tab_sysrisk:
    st.subheader("🧨 Скан системного риска по всей базе")
    col1, col2, col3, col4 = st.columns([1,1,1,1.2])
    with col1:
        sr_threshold = st.slider("Порог риска (%)", 5, 95, 50, 5,
                                 help="Компания B считается поражённой, если её риск (rel_loss%) > порога.")
    with col2:
        sr_max_hops = st.number_input("Макс. волн для симуляции", 1, 8, 3, 1,
                                      help="Глубина каскада для точной оценки Top-K.")
    with col3:
        sr_min_exposure = st.number_input("Мин. экспозиция (MDL)", 0.0, None, float(min_exposure or 0.0), 1000.0,
                                          help="Отсеивает мелкие связи как в основном графе.")
    with col4:
        sr_topk = st.number_input("Top-K для точной симуляции", 5, 100, 20, 5,
                                  help="Для этих компаний запустим каскадную симуляцию на K-хоповом подграфе.")

    sr_types = st.multiselect("Типы связей (фильтр при сканировании)",
                              options=["loan", "trade_credit", "partnership", "ownership"],
                              default=type_filters or [])

    run_scan = st.button("Запустить скан", type="primary")

    @st.cache_data(show_spinner=True)
    def fast_scan_first_wave(threshold_pct: float,
                             min_exposure_mdl: float,
                             type_filters_list: list[str]) -> pd.DataFrame:
        """Быстрый 1-волновый скан по всей базе (без каскада)."""
        conn = open_conn()
        try:
            where_parts = []
            params = []
            if min_exposure_mdl and min_exposure_mdl > 0:
                where_parts.append("r.exposure_mdl >= ?")
                params.append(float(min_exposure_mdl))
            if type_filters_list:
                qmarks = ",".join(["?"] * len(type_filters_list))
                where_parts.append(f"lower(r.relation_type) IN ({qmarks})")
                params.extend([t.lower() for t in type_filters_list])
            where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

            # тянем все рёбра + equity целей
            sql = f"""
                SELECT
                    r.source_company_id AS source,
                    r.target_company_id AS target,
                    r.exposure_mdl      AS exposure,
                    r.lgd_assumption    AS lgd,
                    t.equity_mdl        AS target_equity
                FROM {tbl_rel} r
                JOIN {tbl_comp} t ON t.company_id = r.target_company_id
                {where_sql};
            """
            df = pd.read_sql_query(sql, conn, params=params)

            if df.empty:
                return pd.DataFrame(columns=[
                    "source_id","out_degree","exposure_sum","exp_weighted_loss",
                    "impacted_cnt_first_wave","impacted_loss_first_wave","median_risk_pct"
                ])

            df["lgd"] = df["lgd"].fillna(0.0).clip(0, 1)
            df["loss"] = df["exposure"] * df["lgd"]
            df["risk_pct"] = (df["loss"] / df["target_equity"].replace(0, pd.NA)) * 100.0
            df["risk_pct"] = df["risk_pct"].fillna(0.0).clip(0, None)
            df["is_impacted"] = df["risk_pct"] > float(threshold_pct)

            grp = df.groupby("source", as_index=False).agg(
                out_degree = ("target", "nunique"),
                exposure_sum = ("exposure", "sum"),
                exp_weighted_loss = ("loss", "sum"),
                impacted_cnt_first_wave = ("is_impacted", "sum"),
                impacted_loss_first_wave = ("loss", lambda x: x[df.loc[x.index, "is_impacted"]].sum()),
                median_risk_pct = ("risk_pct", "median"),
            ).rename(columns={"source":"source_id"}).sort_values(
                ["impacted_cnt_first_wave","impacted_loss_first_wave","exp_weighted_loss"],
                ascending=[False, False, False]
            )

            # добавим имя компании
            names = pd.read_sql_query(f"SELECT company_id, company_name FROM {tbl_comp};", conn)
            grp = grp.merge(names, left_on="source_id", right_on="company_id", how="left").drop(columns=["company_id"])
            return grp
        finally:
            conn.close()

    def precise_simulate_for_candidates(candidates: pd.DataFrame,
                                        threshold_pct: float,
                                        max_hops_local: int,
                                        min_exposure_mdl: float,
                                        type_filters_list: list[str],
                                        topk: int) -> pd.DataFrame:
        """Точная каскадная симуляция для Top-K кандидатов (по 1-волновому рангу)."""
        rows = []
        pick = candidates.head(int(topk)).copy()
        for _, row in pick.iterrows():
            src = row["source_id"]
            # собираем K-хоповый подграф от источника
            nodes_k, edges_k = get_out_subgraph_khops(
                src, max_hops_local, min_exposure_mdl, type_filters_list
            )
            if edges_k is None or edges_k.empty:
                continue
            Gx = _build_nx_graph(src, nodes_k, edges_k)
            infected_order, loss_table = simulate_contagion(
                Gx, start_node=src, threshold=threshold_pct/100.0, max_hops=int(max_hops_local)
            )
            infected_cnt = max(0, len(infected_order) - 1)  # без стартового
            total_loss = float(loss_table["abs_loss_mdl"].sum()) if loss_table is not None and not loss_table.empty else 0.0
            max_wave = max(infected_order.values()) if infected_order else 0
            # нормируем к капиталу подграфа (для относительной оценки)
            sub_equity_sum = float(nodes_k["equity_mdl"].fillna(0).sum() or 0.0)
            loss_pct_of_sub = (total_loss / sub_equity_sum * 100.0) if sub_equity_sum > 0 else 0.0

            rows.append({
                "source_id": src,
                "company_name": row.get("company_name",""),
                "infected_cnt_total": infected_cnt,
                "total_abs_loss_mdl": int(round(total_loss)),
                "max_wave": int(max_wave),
                "loss_pct_of_subgraph": round(loss_pct_of_sub, 2),
                "first_wave_impacted_cnt": int(row.get("impacted_cnt_first_wave", 0)),
                "first_wave_impacted_loss": int(round(row.get("impacted_loss_first_wave", 0))),
            })
        if not rows:
            return pd.DataFrame(columns=[
                "source_id","company_name","infected_cnt_total","total_abs_loss_mdl",
                "max_wave","loss_pct_of_subgraph","first_wave_impacted_cnt","first_wave_impacted_loss"
            ])
        out = pd.DataFrame(rows).sort_values(
            ["infected_cnt_total","total_abs_loss_mdl","max_wave"],
            ascending=[False, False, False]
        )
        return out

    if run_scan:
        with st.spinner("Сканирую первую волну по всей базе…"):
            fast = fast_scan_first_wave(sr_threshold, sr_min_exposure, sr_types)

        if fast.empty:
            st.warning("Не удалось построить первичную оценку — проверь фильтры.")
        else:
            show_fast = fast.rename(columns={
                "source_id": "Компания ID",
                "company_name": "Компания",
                "out_degree": "Исходящих связей",
                "exposure_sum": "Сумма экспозиций",
                "exp_weighted_loss": "Ожидаемые потери (все)",
                "impacted_cnt_first_wave": "Заражённых (1-я волна)",
                "impacted_loss_first_wave": "Потери на 1-й волне",
                "median_risk_pct": "Медианный риск (%)",
            })
            st.markdown("#### ⚡ Быстрый скан (1-я волна, без каскада)")
            st.dataframe(show_fast.head(200), use_container_width=True, height=420)

            with st.spinner(f"Считаю каскад для Top-{int(sr_topk)} кандидатов…"):
                precise = precise_simulate_for_candidates(
                    fast, sr_threshold, sr_max_hops, sr_min_exposure, sr_types, sr_topk
                )

            st.markdown("#### 🎯 Точный пересчёт для Top-K")
            if precise.empty:
                st.info("Для выбранных условий каскадная симуляция не дала заражений.")
            else:
                show_prec = precise.rename(columns={
                    "source_id": "Компания ID",
                    "company_name": "Компания",
                    "infected_cnt_total": "Всего заражённых (без источника)",
                    "total_abs_loss_mdl": "Суммарные потери (MDL)",
                    "max_wave": "Макс. волна",
                    "loss_pct_of_subgraph": "Потери от капитала подграфа (%)",
                    "first_wave_impacted_cnt": "Заражённых (1-я волна)",
                    "first_wave_impacted_loss": "Потери на 1-й волне (MDL)",
                })
                st.dataframe(show_prec, use_container_width=True, height=420)

            st.caption(
                "Методика: быстрый скан считает риск у контрагентов от падения источника (1-я волна). "
                "Точный пересчёт строит K-хоповый подграф и моделирует каскад по текущей модели contagion."
            )
