from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
from datetime import datetime
import json

app = Flask(__name__)

# Функция для подключения к базе данных
def get_db_connection():
    conn = sqlite3.connect('corporate_network.db')
    conn.row_factory = sqlite3.Row
    return conn

# Главная страница
@app.route('/')
def index():
    return render_template('index.html')

# API для поиска компании
@app.route('/search_companies')
def search_companies():
    query = request.args.get('q', '')
    conn = get_db_connection()
    
    companies = conn.execute('''
        SELECT company_id, company_name 
        FROM Companii 
        WHERE company_id LIKE ? OR company_name LIKE ?
        LIMIT 10
    ''', (f'%{query}%', f'%{query}%')).fetchall()
    
    conn.close()
    
    results = [{'id': row['company_id'], 'name': row['company_name']} for row in companies]
    return jsonify(results)

# API для получения данных компании
@app.route('/company/<company_id>')
def get_company_data(company_id):
    conn = get_db_connection()
    
    # 1. Основные данные компании
    company_data = conn.execute('''
        SELECT * FROM Companii WHERE company_id = ?
    ''', (company_id,)).fetchone()
    
    if not company_data:
        conn.close()
        return jsonify({'error': 'Company not found'}), 404
    
    # 2. Исходящие связи (компания является источником)
    outgoing_relationships = conn.execute('''
        SELECT 
            c.company_name as target_company_name,
            r.relation_type,
            r.exposure_mdl,
            r.ownership_pct,
            r.start_date,
            r.maturity_days,
            r.loss_impact_on_source_equity
        FROM Relatii r
        JOIN Companii c ON r.target_company_id = c.company_id
        WHERE r.source_company_id = ?
    ''', (company_id,)).fetchall()
    
    # 3. Входящие связи (компания является целью)
    incoming_relationships = conn.execute('''
        SELECT 
            c.company_name as source_company_name,
            r.relation_type,
            r.exposure_mdl,
            r.ownership_pct,
            r.start_date,
            r.maturity_days,
            r.loss_impact_on_source_equity
        FROM Relatii r
        JOIN Companii c ON r.source_company_id = c.company_id
        WHERE r.target_company_id = ?
    ''', (company_id,)).fetchall()
    
    # 4. Анализ риска заражения (contagion analysis)
    # Компании, которые пострадают если наша компания обанкротится
    contagion_impact = conn.execute('''
        SELECT 
            c.company_name,
            r.relation_type,
            r.exposure_mdl as potential_loss,
            r.loss_impact_on_source_equity,
            (r.exposure_mdl / c.equity_mdl * 100) as equity_impact_pct,
            c.equity_mdl,
            c.distress_flag
        FROM Relatii r
        JOIN Companii c ON r.target_company_id = c.company_id
        WHERE r.source_company_id = ? AND r.exposure_mdl > 0
        ORDER BY r.exposure_mdl DESC
    ''', (company_id,)).fetchall()
    
    # 5. Уязвимость компании (от кого может пострадать)
    vulnerability_analysis = conn.execute('''
        SELECT 
            c.company_name,
            r.relation_type,
            r.exposure_mdl as exposure_to_risk,
            r.loss_impact_on_source_equity,
            (r.exposure_mdl / ? * 100) as impact_on_our_equity_pct,
            c.distress_flag,
            c.probability_of_distress
        FROM Relatii r
        JOIN Companii c ON r.source_company_id = c.company_id
        WHERE r.target_company_id = ? AND r.exposure_mdl > 0
        ORDER BY r.exposure_mdl DESC
    ''', (company_data['equity_mdl'] if company_data['equity_mdl'] else 1, company_id)).fetchall()
    
    conn.close()
    
    # Форматируем данные для ответа
    response = {
        'company_data': dict(company_data),
        'outgoing_relationships': [dict(row) for row in outgoing_relationships],
        'incoming_relationships': [dict(row) for row in incoming_relationships],
        'contagion_impact': [dict(row) for row in contagion_impact],
        'vulnerability_analysis': [dict(row) for row in vulnerability_analysis]
    }
    
    return jsonify(response)

# API для расширенного анализа разрушения
@app.route('/contagion_analysis/<company_id>')
def contagion_analysis(company_id):
    conn = get_db_connection()
    
    # Детальный анализ каскадного эффекта
    cascade_effect = conn.execute('''
        WITH RECURSIVE contagion_path AS (
            -- Первый уровень: прямые контрагенты
            SELECT 
                r.target_company_id as company_id,
                c.company_name,
                r.exposure_mdl as direct_exposure,
                r.exposure_mdl as total_exposure,
                1 as level,
                r.relation_type,
                c.probability_of_distress,
                c.distress_flag
            FROM Relatii r
            JOIN Companii c ON r.target_company_id = c.company_id
            WHERE r.source_company_id = ?
            
            UNION ALL
            
            -- Последующие уровни: контрагенты контрагентов
            SELECT 
                r2.target_company_id,
                c2.company_name,
                r2.exposure_mdl,
                cp.total_exposure + r2.exposure_mdl,
                cp.level + 1,
                r2.relation_type,
                c2.probability_of_distress,
                c2.distress_flag
            FROM contagion_path cp
            JOIN Relatii r2 ON cp.company_id = r2.source_company_id
            JOIN Companii c2 ON r2.target_company_id = c2.company_id
            WHERE cp.level < 3  -- Ограничиваем глубину анализа
        )
        SELECT * FROM contagion_path 
        ORDER BY level, total_exposure DESC
    ''', (company_id,)).fetchall()
    
    # Суммарный риск
    total_risk = conn.execute('''
        SELECT 
            SUM(r.exposure_mdl) as total_exposure_at_risk,
            COUNT(DISTINCT r.target_company_id) as num_companies_affected,
            AVG(c.probability_of_distress) as avg_distress_probability
        FROM Relatii r
        JOIN Companii c ON r.target_company_id = c.company_id
        WHERE r.source_company_id = ?
    ''', (company_id,)).fetchone()
    
    conn.close()
    
    return jsonify({
        'cascade_effect': [dict(row) for row in cascade_effect],
        'total_risk': dict(total_risk) if total_risk else {}
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)