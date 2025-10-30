// Auto-complete search
document.getElementById('companySearch').addEventListener('input', function(e) {
    const query = e.target.value;
    if (query.length < 2) {
        document.getElementById('searchResults').innerHTML = '';
        return;
    }

    fetch(`/search_companies?q=${encodeURIComponent(query)}`)
        .then(response => response.json())
        .then(data => {
            const resultsContainer = document.getElementById('searchResults');
            resultsContainer.innerHTML = '';
            
            data.forEach(company => {
                const div = document.createElement('div');
                div.className = 'list-group-item list-group-item-action';
                div.innerHTML = `
                    <strong>${company.name}</strong>
                    <br><small class="text-muted">ID: ${company.id}</small>
                `;
                div.addEventListener('click', () => {
                    document.getElementById('companySearch').value = company.name;
                    resultsContainer.innerHTML = '';
                    loadCompanyData(company.id);
                });
                resultsContainer.appendChild(div);
            });
        });
});

function searchCompany() {
    const query = document.getElementById('companySearch').value;
    if (query.trim()) {
        fetch(`/search_companies?q=${encodeURIComponent(query)}`)
            .then(response => response.json())
            .then(data => {
                if (data.length > 0) {
                    loadCompanyData(data[0].id);
                } else {
                    alert('Compania nu a fost găsită!');
                }
            });
    }
}

function loadCompanyData(companyId) {
    document.getElementById('loading').classList.remove('d-none');
    document.getElementById('companyResults').classList.add('d-none');

    fetch(`/company/${companyId}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById('loading').classList.add('d-none');
            document.getElementById('companyResults').classList.remove('d-none');
            
            displayCompanyData(data);
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('loading').classList.add('d-none');
            alert('Eroare la încărcarea datelor companiei!');
        });
}

function displayCompanyData(data) {
    const company = data.company_data;
    
    // Basic company info
    document.getElementById('companyName').textContent = company.company_name;
    document.getElementById('companyId').textContent = `ID: ${company.company_id}`;
    
    // Financial metrics
    const metricsHtml = `
        <div class="col-md-3 financial-metric">
            <div class="metric-value text-primary">${formatNumber(company.assets_mdl)} MDL</div>
            <small>Total Active</small>
        </div>
        <div class="col-md-3 financial-metric">
            <div class="metric-value ${company.equity_mdl < 0 ? 'text-danger' : 'text-success'}">
                ${formatNumber(company.equity_mdl)} MDL
            </div>
            <small>Capital Propriu</small>
        </div>
        <div class="col-md-3 financial-metric">
            <div class="metric-value ${company.probability_of_distress > 0.3 ? 'text-danger' : 'text-success'}">
                ${((company.probability_of_distress || 0) * 100).toFixed(1)}%
            </div>
            <small>Probabilitate Faliment</small>
        </div>
        <div class="col-md-3 financial-metric">
            <div class="metric-value text-info">${formatNumber(company.employees)}</div>
            <small>Angajați</small>
        </div>
    `;
    document.getElementById('companyMetrics').innerHTML = metricsHtml;
    
    // Relationships
    displayRelationships(data.outgoing_relationships, 'outgoingRelationships', 'outgoing');
    displayRelationships(data.incoming_relationships, 'incomingRelationships', 'incoming');
    
    // Risk analysis
    displayRiskAnalysis(data.contagion_impact, 'contagionImpact', 'contagion');
    displayRiskAnalysis(data.vulnerability_analysis, 'vulnerabilityAnalysis', 'vulnerability');
}

function displayRelationships(relationships, containerId, type) {
    const container = document.getElementById(containerId);
    
    if (relationships.length === 0) {
        container.innerHTML = '<p class="text-muted">Nu există relații</p>';
        return;
    }
    
    let html = '';
    relationships.forEach(rel => {
        const exposure = rel.exposure_mdl || 0;
        const riskClass = exposure > 1000000 ? 'risk-high' : exposure > 100000 ? 'risk-medium' : 'risk-low';
        
        html += `
            <div class="border-bottom pb-2 mb-2 ${riskClass} p-2">
                <div class="d-flex justify-content-between">
                    <strong>${type === 'outgoing' ? rel.target_company_name : rel.source_company_name}</strong>
                    <span class="badge bg-primary">${formatNumber(exposure)} MDL</span>
                </div>
                <small class="text-muted">Tip: ${rel.relation_type}</small>
                ${rel.ownership_pct ? `<br><small>Ownership: ${rel.ownership_pct}%</small>` : ''}
            </div>
        `;
    });
    
    container.innerHTML = html;
}

function displayRiskAnalysis(risks, containerId, type) {
    const container = document.getElementById(containerId);
    
    if (risks.length === 0) {
        container.innerHTML = '<p class="text-muted">Nu există riscuri identificate</p>';
        return;
    }
    
    let html = '';
    let totalExposure = 0;
    
    risks.forEach(risk => {
        const exposure = risk.potential_loss || risk.exposure_to_risk || 0;
        totalExposure += exposure;
        
        const impactPct = risk.equity_impact_pct || risk.impact_on_our_equity_pct || 0;
        const riskLevel = impactPct > 10 ? 'high' : impactPct > 5 ? 'medium' : 'low';
        const riskColor = riskLevel === 'high' ? 'danger' : riskLevel === 'medium' ? 'warning' : 'success';
        
        html += `
            <div class="border-bottom pb-2 mb-2 p-2">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <strong>${risk.company_name}</strong>
                        <span class="badge bg-${riskColor} exposure-badge">${impactPct.toFixed(1)}%</span>
                    </div>
                    <span class="badge bg-secondary">${formatNumber(exposure)} MDL</span>
                </div>
                <small class="text-muted">Tip: ${risk.relation_type}</small>
                ${risk.distress_flag ? '<br><span class="badge bg-danger">Risc Înalt</span>' : ''}
            </div>
        `;
    });
    
    // Add total exposure
    html = `
        <div class="alert alert-info">
            <strong>Expunere Totală: ${formatNumber(totalExposure)} MDL</strong>
        </div>
        ${html}
    `;
    
    container.innerHTML = html;
}

function formatNumber(num) {
    if (!num) return '0';
    return new Intl.NumberFormat('ro-RO').format(Math.round(num));
}