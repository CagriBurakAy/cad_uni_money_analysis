import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import os

# Load & prepare data 
def load_data():


    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")  

    income_path = os.path.join(data_dir, "UniversityRevenue.csv")
    expend_path = os.path.join(data_dir, "UniversityExpenditure.csv")

    # Load the data from CSV
    rev = pd.read_csv(income_path)
    exp = pd.read_csv(expend_path)

    # Select relevant columns
    cols = ['Year', 'Institution_name', 'Province Name', 'Column name', 'Value']
    rev = rev[cols].copy()
    exp = exp[cols].copy()

    rev['Value'] = pd.to_numeric(rev['Value'], errors='coerce').fillna(0)
    exp['Value'] = pd.to_numeric(exp['Value'], errors='coerce').fillna(0)

    rev['Year'] = rev['Year'].astype(str).str.strip()
    exp['Year'] = exp['Year'].astype(str).str.strip()
    rev['Year_num'] = rev['Year'].str.slice(0, 4).astype(int)
    exp['Year_num'] = exp['Year'].str.slice(0, 4).astype(int)

    # Convert to billions (Excel values missing last 3 digits)
    rev['Value'] = (rev['Value'] * 1000) / 1e9
    exp['Value'] = (exp['Value'] * 1000) / 1e9

    rev['Province Name'] = rev['Province Name'].str.strip().str.title()
    exp['Province Name'] = exp['Province Name'].str.strip().str.title()
    rev_totals = rev[rev['Column name'].str.contains('Total funds', case=False, na=False)].copy()
    exp_totals = exp[exp['Column name'].str.contains('Total funds', case=False, na=False)].copy()

    provinces = sorted(set(rev['Province Name'].dropna().unique()) | set(exp['Province Name'].dropna().unique()))
    universities = sorted(set(rev['Institution_name'].dropna().unique()).union(set(exp['Institution_name'].dropna().unique())))

    year_map_df = pd.DataFrame({
        'Year_num': sorted(set(rev['Year_num'].unique()).union(set(exp['Year_num'].unique())))
    })
    year_label_map = {}
    for y in year_map_df['Year_num']:
        label = rev.loc[rev['Year_num'] == y, 'Year'].iloc[0] if not rev.loc[rev['Year_num'] == y, 'Year'].empty else \
                exp.loc[exp['Year_num'] == y, 'Year'].iloc[0] if not exp.loc[exp['Year_num'] == y, 'Year'].empty else str(y)
        year_label_map[y] = label

    return rev, exp, rev_totals, exp_totals, provinces, universities, year_label_map

rev_full, exp_full, rev_totals, exp_totals, province_options, university_options, year_label_map = load_data()

app = Dash(__name__)
server = app.server
app.title = "Canadian Universities: Revenue vs Expenditure"

min_year = min(year_label_map.keys())
max_year = max(year_label_map.keys())

app.layout = html.Div([
    html.H2("Canadian Universities: Revenue vs Expenditure", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Province", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='province-dropdown',
            options=[{'label': p, 'value': p} for p in province_options],
            value=[],
            multi=True,
            placeholder="Select one or more provinces (leave empty for all)"
        ),
    ], style={'maxWidth': '400px', 'marginBottom': '16px'}),



    html.Div([
        html.Label("View Mode", style={'fontWeight': 'bold'}),
        dcc.RadioItems(
            id='view-mode',
            options=[
                {'label': 'Aggregate View', 'value': 'aggregate'},
                {'label': 'Per-Province View', 'value': 'per_province'}
            ],
            value='aggregate',
            labelStyle={'display': 'inline-block', 'marginRight': '20px'}
        )
    ], style={'marginBottom': '20px'}),

    dcc.Graph(id='main-trend'),

    html.Div([
        html.Label("Year range", style={'fontWeight': 'bold', 'fontSize': '18px'}),
        dcc.RangeSlider(
            id='year-slider',
            min=min_year,
            max=max_year,
            value=[min_year, max_year],
            step=1,
            allowCross=False,
            tooltip={"placement": "bottom", "always_visible": True},
            marks={y: str(y) for y in year_label_map}
        ),
        html.Div(id='year-range-label', style={'margin': '10px', 'fontStyle': 'italic', 'fontSize': '16px'})
    ], style={
        'margin': '30px 0',
        'padding': '20px',
        'backgroundColor': '#f9f9f9',
        'borderRadius': '8px'
    }),

    html.H3("Revenue Breakdown (Billions CAD, excludes Total funds)"),
    dcc.Graph(id='rev-breakdown'),

    html.H3("Expenditure Breakdown (Billions CAD, excludes Total funds)"),
    dcc.Graph(id='exp-breakdown'),
    html.Div([
        html.Label("University", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='university-dropdown',
            options=[{'label': u, 'value': u} for u in university_options],
            value=[],
            multi=True,
            placeholder="Select one or more universities (leave empty for all)"
        ),
    ], style={'maxWidth': '600px', 'marginBottom': '16px'}),
    html.H3("University-Level Revenue vs Expenditure"),
    dcc.Graph(id='uni-trend'),

    html.H3("University Revenue Breakdown (Billions CAD, excludes Total funds)"),
    dcc.Graph(id='uni-rev-breakdown'),

    html.H3("University Expenditure Breakdown (Billions CAD, excludes Total funds)"),
    dcc.Graph(id='uni-exp-breakdown'),
], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '12px'})

@app.callback(
    Output('year-range-label', 'children'),
    Output('main-trend', 'figure'),
    Output('rev-breakdown', 'figure'),
    Output('exp-breakdown', 'figure'),
    Output('uni-trend', 'figure'),
    Output('uni-rev-breakdown', 'figure'),
    Output('uni-exp-breakdown', 'figure'),
    Input('province-dropdown', 'value'),
    Input('university-dropdown', 'value'),
    Input('year-slider', 'value'),
    Input('view-mode', 'value')
)
def update_dashboard(selected_provinces, selected_universities, year_range, view_mode):
    start_num, end_num = year_range
    is_single_year = (start_num == end_num)

    rev_f = rev_totals.copy()
    exp_f = exp_totals.copy()
    rev_details = rev_full.copy()
    exp_details = exp_full.copy()

    if selected_provinces:
        rev_f = rev_f[rev_f['Province Name'].isin(selected_provinces)]
        exp_f = exp_f[exp_f['Province Name'].isin(selected_provinces)]
        rev_details = rev_details[rev_details['Province Name'].isin(selected_provinces)]
        exp_details = exp_details[exp_details['Province Name'].isin(selected_provinces)]

    if selected_universities:
        rev_details = rev_details[rev_details['Institution_name'].isin(selected_universities)]
        exp_details = exp_details[exp_details['Institution_name'].isin(selected_universities)]

    rev_f = rev_f[(rev_f['Year_num'] >= start_num) & (rev_f['Year_num'] <= end_num)]
    exp_f = exp_f[(exp_f['Year_num'] >= start_num) & (exp_f['Year_num'] <= end_num)]
    rev_details = rev_details[(rev_details['Year_num'] >= start_num) & (rev_details['Year_num'] <= end_num)]
    exp_details = exp_details[(exp_details['Year_num'] >= start_num) & (exp_details['Year_num'] <= end_num)]

    title_suffix = f"{year_label_map[start_num]}" if is_single_year else f"{year_label_map[start_num]} – {year_label_map[end_num]}"
    prov_suffix = "All provinces" if not selected_provinces else ", ".join(selected_provinces)
    main_title = f"Revenue vs Expenditure ({prov_suffix}; {title_suffix})"

    # Main chart
    if view_mode == 'aggregate':
        rev_by_year = rev_f.groupby(['Year_num'], as_index=False)['Value'].sum().rename(columns={'Value': 'Revenue'})
        exp_by_year = exp_f.groupby(['Year_num'], as_index=False)['Value'].sum().rename(columns={'Value': 'Expenditure'})
        merged = pd.merge(rev_by_year, exp_by_year, on='Year_num', how='outer').fillna(0.0)
        merged['NetBalance'] = merged['Revenue'] - merged['Expenditure']

        df_long = merged.melt(
            id_vars=['Year_num', 'NetBalance'],
            value_vars=['Revenue', 'Expenditure'],
            var_name='Type',
            value_name='Billions CAD'
        )

        fig_main = px.line(
            df_long.sort_values('Year_num'),
            x="Year_num",
            y="Billions CAD",
            color="Type",
            markers=True,
            title=main_title,
            labels={"Year_num": "Year", "Billions CAD": "Billions CAD"}
        )
        fig_main.add_bar(
            x=merged['Year_num'],
            y=merged['NetBalance'],
            name='Net Balance',
            marker_color=['green' if val >= 0 else 'red' for val in merged['NetBalance']],
            opacity=0.5,
            hovertemplate='Year=%{x}<br>Net Balance=%{y:.1f} B CAD<extra></extra>'
        )
        fig_main.update_layout(barmode='overlay',
            xaxis=dict(
                type='category',
                tickmode='linear',
                tick0=start_num,
                dtick=1
    ))

    else:
        rev_by_prov = rev_f.groupby(['Province Name', 'Year_num'], as_index=False)['Value'].sum()
        exp_by_prov = exp_f.groupby(['Province Name', 'Year_num'], as_index=False)['Value'].sum()

        rev_by_prov['Type'] = 'Revenue'
        exp_by_prov['Type'] = 'Expenditure'
        rev_by_prov.rename(columns={'Value': 'Billions CAD'}, inplace=True)
        exp_by_prov.rename(columns={'Value': 'Billions CAD'}, inplace=True)

        df_prov = pd.concat([rev_by_prov, exp_by_prov], axis=0)
        df_prov.sort_values(['Province Name', 'Year_num'], inplace=True)

        fig_main = px.line(
            df_prov,
            x='Year_num',
            y='Billions CAD',
            color='Province Name',
            line_dash='Type',
            markers=True,
            title=main_title,
            labels={'Year_num': 'Year', 'Billions CAD': 'Billions CAD'}
        )
        fig_main.update_layout(
            hovermode="x unified",
            yaxis_title="Billions CAD",
            xaxis=dict(
                type='category',  
                tickmode='linear',
                tick0=start_num,
                dtick=1
            )
        )

    # Revenue Breakdown
    rev_details = rev_details[~rev_details['Column name'].str.contains('Total funds', case=False, na=False)]
    rev_cat = rev_details.groupby('Column name', as_index=False)['Value'].sum().sort_values('Value', ascending=False)

    # Clean the category names
    rev_cat['Category'] = rev_cat['Column name'].str.replace(r'^\d+\.\s*', '', regex=True)

    fig_rev = px.bar(
        rev_cat,
        x='Category',
        y='Value',
        title=f"Revenue Breakdown – {title_suffix}",
        labels={'Value': 'Billions CAD', 'Column name': 'Category'}
    )
    fig_rev.update_layout(
        xaxis_tickangle=-40,
        yaxis_title="Billions CAD"
        
    )

    # Expenditure Breakdown
    exp_details = exp_details[~exp_details['Column name'].str.contains('Total funds', case=False, na=False)]
    exp_cat = exp_details.groupby('Column name', as_index=False)['Value'].sum().sort_values('Value', ascending=False)

    # Clean the category names
    exp_cat['Category'] = exp_cat['Column name'].str.replace(r'^\d+\.\s*', '', regex=True)
    fig_exp = px.bar(
        exp_cat,
        x='Category',
        y='Value',
        title=f"Expenditure Breakdown – {title_suffix}",
        labels={'Value': 'Billions CAD', 'Column name': 'Category'}
    )
    fig_exp.update_layout(
        xaxis_tickangle=-40,
        yaxis_title="Billions CAD"
    )

    # University-Level Chart
    uni_rev = rev_details.groupby(['Institution_name', 'Year_num'], as_index=False)['Value'].sum()
    uni_exp = exp_details.groupby(['Institution_name', 'Year_num'], as_index=False)['Value'].sum()

    uni_rev['Type'] = 'Revenue'
    uni_exp['Type'] = 'Expenditure'
    uni_rev.rename(columns={'Value': 'Billions CAD'}, inplace=True)
    uni_exp.rename(columns={'Value': 'Billions CAD'}, inplace=True)

    df_uni = pd.concat([uni_rev, uni_exp], axis=0)
    df_uni.sort_values(['Institution_name', 'Year_num'], inplace=True)

    fig_uni = px.line(
        df_uni,
        x='Year_num',
        y='Billions CAD',
        color='Institution_name',
        line_dash='Type',
        markers=True,
        title=f"University-Level Revenue vs Expenditure ({title_suffix})",
        labels={'Year_num': 'Year', 'Institution_name': 'University'}
    )
    fig_uni.update_layout(
        hovermode="x unified",
        yaxis_title="Billions CAD",
        xaxis=dict(
                type='category',
                tickmode='linear',
                tick0=start_num,
                dtick=1
    )
    )

    # University Revenue Breakdown
    uni_rev_cat = rev_details.groupby(['Institution_name', 'Column name'], as_index=False)['Value'].sum()
    
    uni_rev_cat['Category'] = uni_rev_cat['Column name'].str.replace(r'^\d+\.\s*', '', regex=True)
    fig_uni_rev = px.bar(
        uni_rev_cat,
        x='Category',
        y='Value',
        color='Institution_name',
        title=f"University Revenue Breakdown – {title_suffix}",
        labels={'Value': 'Billions CAD', 'Column name': 'Category'}
    )
    fig_uni_rev.update_layout(
        xaxis_tickangle=-40,
        yaxis_title="Billions CAD"
    )

    # University Expenditure Breakdown
    uni_exp_cat = exp_details.groupby(['Institution_name', 'Column name'], as_index=False)['Value'].sum()
    
    uni_exp_cat['Category'] = uni_exp_cat['Column name'].str.replace(r'^\d+\.\s*', '', regex=True)
    fig_uni_exp = px.bar(
        uni_exp_cat,
        x='Category',
        y='Value',
        color='Institution_name',
        title=f"University Expenditure Breakdown – {title_suffix}",
        labels={'Value': 'Billions CAD', 'Column name': 'Category'}
    )
    fig_uni_exp.update_layout(
        xaxis_tickangle=-40,
        yaxis_title="Billions CAD"
    )

    label_text = f"Selected: {title_suffix} | Provinces: {prov_suffix}"
    return label_text, fig_main, fig_rev, fig_exp, fig_uni, fig_uni_rev, fig_uni_exp

if __name__ == "__main__":
    app.run(debug=True)
