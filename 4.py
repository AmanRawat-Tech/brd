import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Meter Installation Scheduler",
    layout="wide"
)

# Hardcoded data (for demonstration; users can upload custom data)
data = pd.read_csv('data.csv')
all_holidays = pd.read_csv('all_holidays.csv')
slow_days_df = pd.DataFrame(columns=['slow_days'])

# Scheduling function (adapted from provided code)
def bell_curve_meters_schedule_holiday_cap_all(sub_mru_df, holiday_df, slow_days_df, start_date, end_date,
                                              base_productivity=7, ramp_up_period=15, saturation_threshold=0.85,
                                              saturation_factor=0.7, holiday_factor=0.1, slow_period_factor=0.7,
                                              ramp_up_factor=0.5, ramp_down_factor=0.7, is_slow_day='No'):
    """
    Generate subdivision schedule with bell-shaped meter distribution and holiday/installer caps.
    """
    # Filter by selected subdivisions
    selected_subdivs = st.session_state.get('selected_subdivs', sub_mru_df['SubDiv'].unique())
    sub_mru_df = sub_mru_df[sub_mru_df['SubDiv'].isin(selected_subdivs)]
    
    if sub_mru_df.empty:
        raise ValueError("No data available for the selected subdivisions")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    holidays = pd.to_datetime(holiday_df['holidays']).dt.date.values if holiday_df is not None else []
    slow_days = pd.to_datetime(slow_days_df['slow_days']).dt.date.values if slow_days_df is not None else []
    
    total_days = (end_date - start_date).days + 1
    if total_days <= 0:
        raise ValueError("End date must be after start date")
    
    ramp_up_end = ramp_up_period / 100
    ramp_down_start = saturation_threshold
    
    result_columns = ['Date', 'SubDiv', 'SubDivCode', 'Total_Meters', 'Meters_Installed_Today',
                     'Cumulative_Meters_Installed', 'Installers', 'Productivity', 'Phase', 'Notes']
    all_results = []
    
    for _, row in sub_mru_df.iterrows():
        subdiv = row['SubDiv']
        subdiv_code = row['SubDivCode']
        total_meters = row['Total_Meters']
        installed = row['Meters_Installed_Already']
        remaining = total_meters - installed
        
        x = np.linspace(0, 1, total_days)
        mean = 0.4
        std_dev = 0.20
        bell_curve = np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))
        bell_curve /= bell_curve.sum()
        meter_distribution = (bell_curve * remaining).round().astype(int)
        
        # Adjust for exact remaining meters
        diff = remaining - meter_distribution.sum()
        if diff > 0:
            highest_days = np.argsort(bell_curve)[-int(diff):]
            meter_distribution[highest_days] += 1
        elif diff < 0:
            lowest_days = np.argsort(bell_curve)[:int(abs(diff))]
            meter_distribution[lowest_days] -= 1
            meter_distribution = np.maximum(meter_distribution, 0)
        
        current_date = start_date
        cumulative = installed
        prev_installers = None
        carryover = 0
        
        for d in range(total_days):
            if cumulative >= total_meters:
                break
            meters_target = meter_distribution[d] + carryover
            if meters_target <= 0:
                current_date += pd.Timedelta(days=1)
                continue
            
            timeline_pct = d / total_days
            if timeline_pct < ramp_up_end:
                phase = 'Ramp-up'
                phase_factor = ramp_up_factor + (1 - ramp_up_factor) * (timeline_pct / ramp_up_end)
            elif timeline_pct > ramp_down_start:
                phase = 'Ramp-down'
                phase_factor = ramp_down_factor
            else:
                phase = 'Peak Execution'
                phase_factor = 1.0
            
            is_holiday = current_date.date() in holidays
            is_slow_day_flag = (is_slow_day.lower() == 'yes') or (current_date.date() in slow_days)
            day_factor = holiday_factor if is_holiday else (slow_period_factor if is_slow_day_flag else 1.0)
            actual_productivity = max(1, round(base_productivity * phase_factor * day_factor))
            
            if is_holiday and prev_installers is not None:
                installers_needed = prev_installers
            else:
                installers_needed = max(1, int(np.ceil(meters_target / actual_productivity)))
            
            meters_today = min(installers_needed * actual_productivity, total_meters - cumulative)
            carryover = meters_target - meters_today if meters_today < meters_target else 0
            cumulative += meters_today
            
            notes = []
            if phase != 'Peak Execution':
                notes.append(phase)
            if is_holiday:
                notes.append('Holiday')
            if is_slow_day_flag:
                notes.append('Slow day')
            
            all_results.append({
                'Date': current_date,
                'SubDiv': subdiv,
                'SubDivCode': subdiv_code,
                'Total_Meters': total_meters,
                'Meters_Installed_Today': meters_today,
                'Cumulative_Meters_Installed': cumulative,
                'Installers': installers_needed,
                'Productivity': actual_productivity,
                'Phase': phase,
                'Notes': ', '.join(notes) if notes else None
            })
            prev_installers = installers_needed
            current_date += pd.Timedelta(days=1)
    
    return pd.DataFrame(all_results)

# Plotting functions (converted to Plotly)
def plot_installation_curve(schedule_df, selected_subdivs=None):
    df = schedule_df.copy()
    if selected_subdivs is None:
        selected_subdivs = df['SubDiv'].unique()
    elif isinstance(selected_subdivs, str):
        selected_subdivs = [selected_subdivs]
    
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, subdiv in enumerate(selected_subdivs):
        subdiv_data = df[df['SubDiv'] == subdiv]
        max_install = subdiv_data['Meters_Installed_Today'].max()
        max_installers = subdiv_data['Installers'].max()
        scaling_factor = max_install / max_installers if max_installers > 0 else 1
        
        fig.add_trace(go.Scatter(
            x=subdiv_data['Date'],
            y=subdiv_data['Meters_Installed_Today'],
            name=f'{subdiv} - Daily',
            mode='lines+markers',
            marker=dict(size=6, color=colors[i % len(colors)]),
            line=dict(width=2, color=colors[i % len(colors)])
        ))
        fig.add_trace(go.Scatter(
            x=subdiv_data['Date'],
            y=subdiv_data['Installers'] * scaling_factor,
            name=f'{subdiv} - Installers (scaled)',
            mode='lines',
            line=dict(width=2, dash='dash', color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title=dict(text='Bell-Shaped Installation Curve', x=0.5, xanchor='center'),
        xaxis_title='Date',
        yaxis_title='Meters Installed / Scaled Installers',
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top'),
        template='plotly_white',
        height=600,
        margin=dict(t=80, b=50, l=50, r=50),
        hovermode='x unified'
    )
    return fig

def plot_monthly_installation(schedule_df, selected_subdivs=None):
    df = schedule_df.copy()
    df['Month'] = df['Date'].dt.to_period('M')
    df['Month_Name'] = df['Date'].dt.strftime('%b-%Y')
    
    if selected_subdivs is None:
        selected_subdivs = df['SubDiv'].unique()
    elif isinstance(selected_subdivs, str):
        selected_subdivs = [selected_subdivs]
    
    monthly_data = df[df['SubDiv'].isin(selected_subdivs)].groupby(
        ['Month', 'Month_Name', 'SubDiv'])['Meters_Installed_Today'].sum().reset_index()
    monthly_data = monthly_data.sort_values('Month').reset_index(drop=True)
    
    fig = px.bar(
        monthly_data,
        x='Month_Name',
        y='Meters_Installed_Today',
        color='SubDiv',
        barmode='group',
        title='Monthly Meter Installation',
        labels={'Month_Name': 'Month', 'Meters_Installed_Today': 'Meters Installed'}
    )
    fig.update_layout(
        title=dict(text='Monthly Meter Installation', x=0.5, xanchor='center'),
        template='plotly_white',
        height=600,
        margin=dict(t=80, b=50, l=50, r=50),
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top'),
        xaxis=dict(tickangle=45),
        hovermode='x unified'
    )
    return fig

def plot_installer_requirements(schedule_df, selected_subdivs=None):
    df = schedule_df.copy()
    df['Month'] = df['Date'].dt.to_period('M')
    df['Month_Name'] = df['Date'].dt.strftime('%b-%Y')
    
    if selected_subdivs is None:
        selected_subdivs = df['SubDiv'].unique()
    elif isinstance(selected_subdivs, str):
        selected_subdivs = [selected_subdivs]
    
    monthly_installers = df[df['SubDiv'].isin(selected_subdivs)].groupby(
        ['Month', 'Month_Name', 'SubDiv']).agg({
            'Installers': ['mean', 'max']
        }).reset_index()
    monthly_installers.columns = ['Month', 'Month_Name', 'SubDiv', 'Avg_Installers', 'Max_Installers']
    monthly_installers = monthly_installers.sort_values('Month').reset_index(drop=True)
    
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, subdiv in enumerate(selected_subdivs):
        sub_data = monthly_installers[monthly_installers['SubDiv'] == subdiv]
        fig.add_trace(go.Scatter(
            x=sub_data['Month_Name'],
            y=sub_data['Avg_Installers'],
            name=f'{subdiv} - Avg Daily',
            mode='lines+markers',
            marker=dict(size=8, color=colors[i % len(colors)]),
            line=dict(width=2, color=colors[i % len(colors)])
        ))
        fig.add_trace(go.Scatter(
            x=sub_data['Month_Name'],
            y=sub_data['Max_Installers'],
            name=f'{subdiv} - Peak Daily',
            mode='lines+markers',
            marker=dict(size=8, color=colors[(i + 1) % len(colors)]),
            line=dict(width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=dict(text='Monthly Installer Requirements', x=0.5, xanchor='center'),
        xaxis_title='Month',
        yaxis_title='Installers',
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top'),
        template='plotly_white',
        height=600,
        margin=dict(t=80, b=50, l=50, r=50),
        xaxis=dict(tickangle=45),
        yaxis=dict(gridcolor='lightgray'),
        hovermode='x unified'
    )
    return fig

def plot_cumulative_progress(schedule_df, selected_subdivs=None):
    df = schedule_df.copy()
    if selected_subdivs is None:
        selected_subdivs = df['SubDiv'].unique()
    elif isinstance(selected_subdivs, str):
        selected_subdivs = [selected_subdivs]
    
    df = df[df['SubDiv'].isin(selected_subdivs)]
    
    # Cumulative Meters Plot
    fig1 = go.Figure()
    colors = px.colors.qualitative.Plotly
    # for i croire, subdiv in enumerate(selected_subdivs):

    #     subdiv_data = df[df['SubDiv'] == subdiv]
    #     fig1.add_trace(go.Scatter(
    #         x=subdiv_data['Date'],
    #         y=subdiv_data['Cumulative_Meters_Installed'],
    #         name=subdiv,
    #         mode='lines+markers',
    #         marker=dict(size=6, color=colors[i % len(colors)]),
    #         line=dict(width=2, color=colors[i % len(colors)])
    #     ))
    for i, subdiv in enumerate(selected_subdivs):
        subdiv_data = df[df['SubDiv'] == subdiv]
        fig1.add_trace(go.Scatter(
            x=subdiv_data['Date'],
            y=subdiv_data['Cumulative_Meters_Installed'],
            name=subdiv,
            mode='lines+markers',
            marker=dict(size=6, color=colors[i % len(colors)]),
            line=dict(width=2, color=colors[i % len(colors)])
        ))

    
    # Progress Percentage Plot
    fig2 = go.Figure()
    for i, subdiv in enumerate(selected_subdivs):
        subdiv_data = df[df['SubDiv'] == subdiv]
        progress_pct = (subdiv_data['Cumulative_Meters_Installed'] / subdiv_data['Total_Meters']) * 100
        fig2.add_trace(go.Scatter(
            x=subdiv_data['Date'],
            y=progress_pct,
            name=subdiv,
            mode='lines+markers',
            marker=dict(size=6, color=colors[i % len(colors)]),
            line=dict(width=2, color=colors[i % len(colors)])
        ))
    
    fig1.update_layout(
        title=dict(text='Cumulative Meters Installed Over Time', x=0.5, xanchor='center'),
        xaxis_title='Date',
        yaxis_title='Cumulative Meters',
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top'),
        template='plotly_white',
        height=500,
        margin=dict(t=80, b=50, l=50, r=50),
        hovermode='x unified'
    )
    
    fig2.update_layout(
        title=dict(text='Installation Progress Percentage', x=0.5, xanchor='center'),
        xaxis_title='Date',
        yaxis_title='Progress (%)',
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top'),
        template='plotly_white',
        height=500,
        margin=dict(t=80, b=50, l=50, r=50),
        hovermode='x unified'
    )
    
    return [fig1, fig2]

# Sidebar rendering
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration", divider='gray')
        
        with st.expander("📍 Subdivision Selection", expanded=True):
            st.subheader("Select Subdivisions")
            all_subdivs = st.session_state.data['SubDiv'].unique()
            select_all = st.checkbox(
                "Select All Subdivisions",
                value=st.session_state.get('is_select_all', False),
                key="select_all_subdivs",
                help="Check to select all subdivisions at once"
            )
            if select_all:
                st.session_state.selected_subdivs = list(all_subdivs)
                st.session_state.is_select_all = True
            else:
                st.session_state.selected_subdivs = st.multiselect(
                    "Choose Subdivisions (one or more):",
                    options=all_subdivs,
                    default=st.session_state.selected_subdivs,
                    key="global_subdiv_select",
                    help="Select one or more subdivisions to include in scheduling and visualizations"
                )
                st.session_state.is_select_all = len(st.session_state.selected_subdivs) == len(all_subdivs)
        
        with st.expander("📁 Data Selection", expanded=True):
            st.subheader("Subdivision Data")
            use_default_data = st.checkbox("Use Default Subdivision Data", value=True, key="data_check")
            if not use_default_data:
                data_file = st.file_uploader("Upload Subdivision Data (CSV)", type="csv", key="data_file")
                if data_file is not None:
                    st.session_state.data = pd.read_csv(data_file)
                    st.session_state.selected_subdivs = []
                    st.session_state.is_select_all = False
            else:
                st.session_state.data = data

            st.subheader("Holidays")
            use_default_holidays = st.checkbox("Use Default Holidays", value=True, key="holidays_check")
            if not use_default_holidays:
                holidays_file = st.file_uploader("Upload Holidays (CSV)", type="csv", key="holidays_file")
                if holidays_file is not None:
                    st.session_state.all_holidays = pd.read_csv(holidays_file, header=None, names=['holidays'])
            else:
                st.session_state.all_holidays = all_holidays

            st.subheader("Slow Days")
            use_default_slow_days = st.checkbox("Use Empty Slow Days", value=True, key="slow_days_check")
            if not use_default_slow_days:
                slow_days_input = st.text_area("Enter slow days (YYYY-MM-DD, one per line)", height=100, key="slow_days_input")
                slow_days_list = [date.strip() for date in slow_days_input.split("\n") if date.strip()]
                st.session_state.slow_days_df = pd.DataFrame(slow_days_list, columns=['slow_days']) if slow_days_list else pd.DataFrame(columns=['slow_days'])
            else:
                st.session_state.slow_days_df = slow_days_df

        with st.expander("📅 Scheduling Parameters", expanded=True):
            st.markdown("**Timeline Settings**")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=st.session_state.get('start_date', datetime(2024, 8, 1)), key="start_date")
            with col2:
                end_date = st.date_input("End Date", value=st.session_state.get('end_date', datetime(2026, 1, 31)), key="end_date")
            
            st.markdown("**Productivity Settings**")
            base_productivity = st.number_input(
                "Base Productivity (meters/installer/day)", 
                value=st.session_state.get('base_productivity', 7.0), 
                min_value=1.0, 
                step=1.0, 
                key="base_productivity"
            )
            
            st.markdown("**Ramp-up Phase**")
            col1, col2 = st.columns(2)
            with col1:
                ramp_up_period = st.slider(
                    "Ramp-up Period (%)", 
                    5, 30, 
                    st.session_state.get('ramp_up_period', 15), 
                    key="ramp_up_period"
                )
            with col2:
                ramp_up_factor = st.slider(
                    "Ramp-up Factor", 
                    0.1, 1.0, 
                    st.session_state.get('ramp_up_factor', 0.5), 
                    step=0.1
                )
            
            st.markdown("**Ramp-down Phase**")
            col1, col2 = st.columns(2)
            with col1:
                saturation_threshold = st.slider(
                    "Saturation Threshold (%)", 
                    50, 95, 
                    int(st.session_state.get('saturation_threshold', 0.85) * 100), 
                    key="saturation_threshold_slider"
                )
            with col2:
                ramp_down_factor = st.slider(
                    "Saturation Factor", 
                    0.1, 1.0, 
                    st.session_state.get('ramp_down_factor', 0.7), 
                    step=0.1
                )
            
            st.markdown("**Special Conditions**")
            holiday_factor = st.slider(
                "Holiday Productivity Factor", 
                0.0, 1.0, 
                st.session_state.get('holiday_factor', 0.1), 
                step=0.05
            )
            slow_period_factor = st.slider(
                "Slow Period Factor", 
                0.1, 1.0, 
                st.session_state.get('slow_period_factor', 0.7), 
                step=0.1
            )
            # is_slow_day = st.selectbox(
            #     "Apply Slow Day to All Days", 
            #     ["No", "Yes"], 
            #     index=0 if st.session_state.get('is_slow_day', 'No') == 'No' else 1
            # )

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = data
if 'all_holidays' not in st.session_state:
    st.session_state.all_holidays = all_holidays
if 'slow_days_df' not in st.session_state:
    st.session_state.slow_days_df = slow_days_df
if 'schedule_df' not in st.session_state:
    st.session_state.schedule_df = None
if 'selected_subdivs' not in st.session_state:
    st.session_state.selected_subdivs = []
if 'is_select_all' not in st.session_state:
    st.session_state.is_select_all = False
if 'selected_plots' not in st.session_state:
    st.session_state.selected_plots = [
        "Bell-Shaped Installation Curve",
        "Monthly Meter Installation",
        "Monthly Installer Requirements",
        "Cumulative Progress"
    ]
if 'start_date' not in st.session_state:
    st.session_state.start_date = datetime(2024, 8, 1)
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime(2026, 1, 31)
if 'base_productivity' not in st.session_state:
    st.session_state.base_productivity = 7.0
if 'ramp_up_period' not in st.session_state:
    st.session_state.ramp_up_period = 15
if 'ramp_up_factor' not in st.session_state:
    st.session_state.ramp_up_factor = 0.5
if 'saturation_threshold' not in st.session_state:
    st.session_state.saturation_threshold = 0.85
if 'ramp_down_factor' not in st.session_state:
    st.session_state.ramp_down_factor = 0.7
if 'holiday_factor' not in st.session_state:
    st.session_state.holiday_factor = 0.1
if 'slow_period_factor' not in st.session_state:
    st.session_state.slow_period_factor = 0.7
if 'is_slow_day' not in st.session_state:
    st.session_state.is_slow_day = 'No'

# Page definitions
def home_page():
    st.title("📊 Meter Installation Scheduler")
    st.markdown("""
        <style>
        .big-font {
            font-size:18px !important;
        }
        </style>
        <div class='big-font'>
        Welcome to the Meter Installation Scheduler! This tool helps you plan and visualize meter installation schedules 
        using a bell-shaped curve approach with holiday and installer constraints.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("📖 How to Use This Tool", expanded=True):
        st.markdown("""
        ### Step-by-Step Guide
        1. **Select Subdivisions**: Choose one or more subdivisions in the sidebar.
        2. **Data Preview**: Verify and edit input data (subdivisions, holidays, slow days).
        3. **Schedule Generation**: Configure parameters and generate the schedule.
        4. **Visual Analysis**: Explore interactive visualizations of the schedule.
        
        ### Key Features
        - **Bell-shaped meter distribution**: Realistic ramp-up and ramp-down phases.
        - **Holiday and slow day adjustments**: Accounts for reduced productivity.
        - **Installer constraints**: Maintains consistent installer counts on holidays.
        - **Interactive visualizations**: Plotly charts for detailed analysis.
        - **Export capabilities**: Download schedules and charts.
        """)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Tip**: Use the sidebar to configure data sources and parameters.")
    with col2:
        st.success("🚀 **Ready?** Select a page from the sidebar to begin!")

def data_preview_page():
    st.title("🔍 Data Preview")
    st.markdown("Review and verify the input data for schedule generation.")
    st.markdown("---")
    
    if not st.session_state.selected_subdivs:
        st.warning("⚠️ Please select at least one subdivision in the sidebar.")
        return
    
    with st.expander("📊 Subdivision Data", expanded=True):
        st.markdown("""
        **Columns**:
        - **SubDiv**: Subdivision name
        - **SubDivCode**: Subdivision code
        - **Total_Meters**: Total meters to install
        - **Meters_Installed_Already**: Meters already installed
        """)
        filtered_data = st.session_state.data[st.session_state.data['SubDiv'].isin(st.session_state.selected_subdivs)]
        edited_data = st.data_editor(filtered_data, num_rows="dynamic", use_container_width=True)
        if st.button("Save Subdivision Data Changes"):
            st.session_state.data = edited_data.copy()
            st.success("Subdivision data updated!")
    
    with st.expander("🎉 Holidays", expanded=True):
        st.markdown("List of dates with reduced productivity due to holidays.")
        edited_holidays = st.data_editor(st.session_state.all_holidays, num_rows="dynamic", use_container_width=True)
        if st.button("Save Holiday Data Changes"):
            st.session_state.all_holidays = edited_holidays
            st.success("Holiday data updated!")
    
    with st.expander("🐢 Slow Days", expanded=True):
        st.markdown("Dates with reduced productivity (e.g., weather or supply issues).")
        if st.session_state.slow_days_df.empty:
            st.info("No slow days currently defined.")
        else:
            edited_slow_days = st.data_editor(st.session_state.slow_days_df, num_rows="dynamic", use_container_width=True)
            if st.button("Save Slow Days Changes"):
                st.session_state.slow_days_df = edited_slow_days
                st.success("Slow days data updated!")
    
    st.markdown("---")
    st.success("✅ Data verified! Proceed to the Schedule page to generate your plan.")

def schedule_page():
    st.title("📅 Generate Schedule")
    st.markdown("""
    Create a meter installation schedule for selected subdivisions using a bell-shaped curve 
    with holiday and installer constraints.
    """)
    st.markdown("---")
    
    if not st.session_state.selected_subdivs:
        st.warning("⚠️ Please select at least one subdivision in the sidebar.")
        return
    
    with st.expander("🔎 Current Parameters Summary", expanded=True):
        col1, col2, col3 = st.columns([3,1,1])
        with col1:
            st.metric("Start Date", st.session_state.start_date.strftime('%Y-%m-%d'))
            st.metric("Base Productivity", f"{st.session_state.base_productivity} meters/installer/day")
            st.metric("Ramp-up Period", f"{st.session_state.ramp_up_period}%")
        with col2:
            st.metric("End Date", st.session_state.end_date.strftime('%Y-%m-%d'))
            st.metric("Saturation Threshold", f"{int(st.session_state.saturation_threshold*100)}%")
            st.metric("Ramp-up Factor", f"{st.session_state.ramp_up_factor*100}%")
        with col3:
            st.metric("Total Days", (st.session_state.end_date - st.session_state.start_date).days + 1)
            st.metric("Ramp-down Factor", f"{st.session_state.ramp_down_factor*100}%")
            st.metric("Holiday Factor", f"{st.session_state.holiday_factor*100}%")
    
    if st.button("✨ Generate Schedule", type="primary", use_container_width=True):
        with st.spinner("Generating schedule..."):
            try:
                st.session_state.schedule_df = bell_curve_meters_schedule_holiday_cap_all(
                    sub_mru_df=st.session_state.data,
                    holiday_df=st.session_state.all_holidays,
                    slow_days_df=st.session_state.slow_days_df,
                    start_date=st.session_state.start_date,
                    end_date=st.session_state.end_date,
                    base_productivity=st.session_state.base_productivity,
                    ramp_up_period=st.session_state.ramp_up_period,
                    saturation_threshold=st.session_state.saturation_threshold,
                    ramp_up_factor=st.session_state.ramp_up_factor,
                    ramp_down_factor=st.session_state.ramp_down_factor,
                    holiday_factor=st.session_state.holiday_factor,
                    slow_period_factor=st.session_state.slow_period_factor,
                    is_slow_day=st.session_state.is_slow_day
                )
                
                st.success("Schedule generated successfully!")
                total_meters = st.session_state.schedule_df.groupby('SubDiv')['Total_Meters'].first().sum()
                installed = st.session_state.schedule_df.groupby('SubDiv')['Cumulative_Meters_Installed'].last().sum()
                completion_percent = (installed / total_meters) * 100
                max_installers = st.session_state.schedule_df['Installers'].max()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div style="font-size: 14px;">
                        <strong>Total Meters</strong><br>
                        <span style="font-size: 18px;">{total_meters:,}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div style="font-size: 14px;">
                        <strong>Will Be Installed</strong><br>
                        <span style="font-size: 18px;">{installed:,} ({completion_percent:.1f}%)</span>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div style="font-size: 14px;">
                        <strong>Peak Installers Needed</strong><br>
                        <span style="font-size: 18px;">{max_installers}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.subheader("Schedule Preview")
                st.dataframe(st.session_state.schedule_df, height=400, use_container_width=True)
                
                csv = st.session_state.schedule_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Schedule as CSV",
                    data=csv,
                    file_name="installation_schedule.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.markdown("---")
                st.success("Navigate to the Visualizations page to explore your schedule graphically!")
                
            except Exception as e:
                st.error(f"❌ Error generating schedule: {str(e)}")
    
    elif st.session_state.schedule_df is not None:
        st.subheader("Existing Schedule")
        st.info("A previously generated schedule exists. Regenerate with new parameters or view visualizations.")
        
        total_meters = st.session_state.schedule_df.groupby('SubDiv')['Total_Meters'].first().sum()
        installed = st.session_state.schedule_df.groupby('SubDiv')['Cumulative_Meters_Installed'].last().sum()
        completion_percent = (installed / total_meters) * 100
        
        col1, col2 = st.columns(2)
        col1.metric("Total Meters in Schedule", f"{total_meters:,}")
        col2.metric("Scheduled for Installation", f"{installed:,} ({completion_percent:.1f}%)")
        
        st.dataframe(st.session_state.schedule_df, height=400, use_container_width=True)
        
        csv = st.session_state.schedule_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Schedule as CSV",
            data=csv,
            file_name="installation_schedule.csv",
            mime="text/csv",
            use_container_width=True
        )

def visualizations_page():
    st.title("📈 Visualizations")
    st.markdown("Interactive visualizations of your meter installation schedule.")
    st.markdown("---")
    
    if st.session_state.schedule_df is None:
        st.warning("⚠️ No schedule found. Generate a schedule on the Schedule page first.")
        return
    
    if not st.session_state.selected_subdivs:
        st.warning("⚠️ Please select at least one subdivision in the sidebar.")
        return
    
    with st.expander("🔧 Visualization Controls", expanded=True):
        st.subheader("Chart Options")
        plot_options = [
            "Bell-Shaped Installation Curve",
            "Monthly Meter Installation",
            "Monthly Installer Requirements",
            "Cumulative Progress"
        ]
        st.session_state.selected_plots = st.multiselect(
            "Select charts to display:",
            options=plot_options,
            default=st.session_state.selected_plots,
            key="plot_select"
        )
    
    st.markdown("---")
    
    for plot in st.session_state.selected_plots:
        with st.container():
            st.subheader(plot)
            
            if plot == "Bell-Shaped Installation Curve":
                st.markdown("""
                **Daily installation progress** showing the bell-shaped curve with:
                - **Solid line**: Meters installed each day
                - **Dashed line**: Installer count (scaled)
                """)
                fig = plot_installation_curve(st.session_state.schedule_df, st.session_state.selected_subdivs)
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot == "Monthly Meter Installation":
                st.markdown("""
                **Monthly totals** showing installation volume by subdivision.
                """)
                fig = plot_monthly_installation(st.session_state.schedule_df, st.session_state.selected_subdivs)
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot == "Monthly Installer Requirements":
                st.markdown("""
                **Average and peak installers needed per day** by month.
                """)
                fig = plot_installer_requirements(st.session_state.schedule_df, st.session_state.selected_subdivs)
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot == "Cumulative Progress":
                st.markdown("""
                **Progress over time** showing cumulative meters installed and percentage completion.
                """)
                figs = plot_cumulative_progress(st.session_state.schedule_df, st.session_state.selected_subdivs)
                for fig in figs:
                    st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("💾 Download This Chart"):
                col1, col2 = st.columns(2)
                with col1:
                    buf = io.BytesIO()
                    fig.write_image(buf, format="png", width=1200, height=500)
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode()
                    href = f'<a href="data:image/png;base64,{b64}" download="{plot.lower().replace(" ", "_")}.png">⬇️ Download as PNG</a>'
                    st.markdown(href, unsafe_allow_html=True)
                with col2:
                    buf = io.BytesIO()
                    fig.write_image(buf, format="svg", width=1200, height=500)
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode()
                    href = f'<a href="data:image/svg+xml;base64,{b64}" download="{plot.lower().replace(" ", "_")}.svg">⬇️ Download as SVG</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            st.markdown("---")

# Define pages
pages = {
    "🏠 Home": home_page,
    "🔍 Data Preview": data_preview_page,
    "📅 Schedule": schedule_page,
    "📈 Visualizations": visualizations_page
}

# Render sidebar and navigation
render_sidebar()
st.sidebar.header("🧭 Navigation", divider='gray')
selection = st.sidebar.radio("Go to", list(pages.keys()), label_visibility="collapsed")
pages[selection]()