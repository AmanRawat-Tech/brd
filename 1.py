import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
#pio.kaleido.scope._DEFAULT_CHROMIUM_EXECUTABLE = "/usr/bin/chromium"
st.set_page_config(
    page_title=" Plan Predictor",
    layout="wide"
)

# Hardcoded data from user input (for demonstration; in practice, these are loaded)
data = pd.read_csv('data.csv')
all_holidays = pd.read_csv('all_holidays.csv')
slow_days_df = pd.DataFrame(columns=['slow_days'])

# Define the scheduling function (modified to filter by selected subdivisions)
def run_base_mru_schedule(sub_mru_data, all_holidays, slow_days_df, start_date, end_date,
                         base_productivity=7, saturation_threshold=0.85,
                         ramp_up_period=15, ramp_up_factor=0.5, ramp_down_factor=0.7,
                         holiday_factor=0.1, slow_period_factor=0.7, is_slow_day='No'):
    """
    Generate subdivision schedule with full-period bell-shaped installation curve
    Parameters:
        ramp_up_period: Percentage of timeline for ramp-up (default: 15)
    """
    # Filter sub_mru_data based on selected subdivisions
    selected_subdivs = st.session_state.get('selected_subdivs', sub_mru_data['SubDiv'].unique())
    sub_mru_data = sub_mru_data[sub_mru_data['SubDiv'].isin(selected_subdivs)]
    
    if sub_mru_data.empty:
        raise ValueError("No data available for the selected subdivisions")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    holidays = pd.to_datetime(all_holidays['holidays']).dt.date.values if all_holidays is not None else []
    slow_days = pd.to_datetime(slow_days_df['slow_days']).dt.date.values if slow_days_df is not None else []
   
    total_days = (end_date - start_date).days + 1
    if total_days <= 0:
        raise ValueError("End date must be after start date")
   
    ramp_up_end = ramp_up_period / 100
    ramp_down_start = saturation_threshold
   
    result_columns = ['Date', 'SubDiv', 'SubDivCode', 'Total_Meters',
                     'Meters_Installed_Today', 'Cumulative_Meters_Installed',
                     'Installers', 'Productivity', 'Phase', 'Notes']
    result_df = pd.DataFrame(columns=result_columns)
   
    for _, subdiv_row in sub_mru_data.iterrows():
        subdiv = subdiv_row['SubDiv']
        subdiv_code = subdiv_row['SubDivCode']
        total_meters = subdiv_row['Total_Meters']
        installed_already = subdiv_row['Meters_Installed_Already']
       
        remaining_meters = total_meters - installed_already
        daily_progress = []
        cumulative_installed = installed_already
       
        x = np.linspace(0, 1, total_days)
        mean = 0.5
        std_dev = 0.15
        bell_curve = np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))
        bell_curve /= bell_curve.max()
       
        theoretical_max_installers = int(np.ceil(remaining_meters / (base_productivity * total_days * 0.5)))
        installer_distribution = (bell_curve * theoretical_max_installers).astype(int)
       
        current_date = start_date
       
        for day_num in range(total_days):
            if cumulative_installed >= total_meters:
                break
               
            installers_today = max(1, installer_distribution[day_num])
           
            timeline_percent = day_num / total_days
            if timeline_percent < ramp_up_end:
                phase = "Ramp-up"
                phase_factor = ramp_up_factor + (1 - ramp_up_factor) * (timeline_percent / ramp_up_end)
            elif timeline_percent > ramp_down_start:
                phase = "Ramp-down"
                phase_factor = ramp_down_factor
            else:
                phase = "Peak Execution"
                phase_factor = 1.0
           
            factors = [phase_factor]
           
            is_holiday = current_date.date() in holidays
            if is_holiday:
                factors.append(holiday_factor)
           
            is_slow_day_flag = (is_slow_day.lower() == 'yes') or (current_date.date() in slow_days)
            if is_slow_day_flag:
                factors.append(slow_period_factor)
           
            final_factor = min(factors)
            productivity_today = round(base_productivity * final_factor)
           
            notes = []
            if timeline_percent < ramp_up_end:
                notes.append(f"Ramp-up ({ramp_up_period}%)")
            elif timeline_percent > ramp_down_start:
                notes.append(f"Ramp-down ({int(ramp_down_start*100)}%)")
            if is_holiday:
                notes.append("Holiday")
            if is_slow_day_flag:
                notes.append("Slow day")
           
            max_possible = installers_today * productivity_today
            meters_today = min(max_possible, remaining_meters)
           
            cumulative_installed += meters_today
            remaining_meters = total_meters - cumulative_installed
           
            daily_progress.append({
                'Date': current_date,
                'SubDiv': subdiv,
                'SubDivCode': subdiv_code,
                'Total_Meters': total_meters,
                'Meters_Installed_Today': meters_today,
                'Cumulative_Meters_Installed': cumulative_installed,
                'Installers': installers_today,
                'Productivity': productivity_today,
                'Phase': phase,
                'Notes': ", ".join(notes) if notes else None
            })
           
            current_date += pd.Timedelta(days=1)
       
        subdiv_df = pd.DataFrame(daily_progress)
        result_df = pd.concat([result_df, subdiv_df], ignore_index=True)
   
    return result_df

# Plotly plotting functions (unchanged, but will use global selected_subdivs)
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
            #line=dict(width=2, dash='dash', color='red')

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
        ['Month', 'Month_Name', 'SubDiv'])['Installers'].mean().reset_index()
    
    fig = px.line(
        monthly_installers,
        x='Month_Name',
        y='Installers',
        color='SubDiv',
        markers=True,
        title='Monthly Installer Requirements',
        labels={'Month_Name': 'Month', 'Installers': 'Average Installers per Day'}
    )
    fig.update_traces(marker=dict(size=8), line=dict(width=2))
    fig.update_layout(
        title=dict(text='Monthly Installer Requirements', x=0.5, xanchor='center'),
        template='plotly_white',
        height=600,
        margin=dict(t=80, b=50, l=50, r=50),
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top'),
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
    
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, subdiv in enumerate(selected_subdivs):
        subdiv_data = df[df['SubDiv'] == subdiv]
        fig.add_trace(go.Scatter(
            x=subdiv_data['Date'],
            y=subdiv_data['Cumulative_Meters_Installed'],
            name=subdiv,
            mode='lines+markers',
            marker=dict(size=6, color=colors[i % len(colors)]),
            line=dict(width=2, color=colors[i % len(colors)])
        ))
    
    final_progress = df.groupby('SubDiv').last().reset_index()
    targets = df.groupby('SubDiv')['Total_Meters'].first().reset_index()
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=final_progress['SubDiv'],
        y=final_progress['Cumulative_Meters_Installed'],
        name='Installed',
        marker_color='rgba(31, 119, 180, 0.8)'
    ))
    fig_bar.add_trace(go.Bar(
        x=targets['SubDiv'],
        y=targets['Total_Meters'],
        name='Target',
        marker_color='rgba(255, 127, 14, 0.3)'
    ))
    
    for i, row in final_progress.iterrows():
        percentage = (row['Cumulative_Meters_Installed'] / row['Total_Meters']) * 100
        fig_bar.add_annotation(
            x=row['SubDiv'],
            y=row['Cumulative_Meters_Installed'] + (row['Total_Meters'] * 0.02),
            text=f"{percentage:.1f}%",
            showarrow=False,
            font=dict(size=10)
        )
    
    fig.update_layout(
        title=dict(text='Cumulative Meters Installed Over Time', x=0.5, xanchor='center'),
        xaxis_title='Date',
        yaxis_title='Cumulative Meters',
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top'),
        template='plotly_white',
        height=500,
        margin=dict(t=80, b=50, l=50, r=50),
        hovermode='x unified'
    )
    
    fig_bar.update_layout(
        title=dict(text='Final Installation Progress vs Target', x=0.5, xanchor='center'),
        xaxis_title='SubDivision',
        yaxis_title='Meters',
        barmode='group',
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top'),
        template='plotly_white',
        height=500,
        margin=dict(t=80, b=50, l=50, r=50),
        yaxis=dict(gridcolor='lightgray'),
        hovermode='x unified'
    )
    
    return [fig, fig_bar]


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration", divider='gray')
        
        with st.expander("📍 Subdivision Selection", expanded=True):
            st.subheader("Select Subdivisions")
            all_subdivs = st.session_state.data['SubDiv'].unique()
            
            # Checkbox to select all subdivisions
            # aman begin
            # ----->>>>>>>*** yaha bhi neeche
            # select_all = st.checkbox(
            #     "Select All Subdivisions",
            #     value=st.session_state.is_select_all,
            #     key="select_all_subdivs",
            #     help="Check to select all subdivisions at once"
            # )
            
            # # Update selected_subdivs based on checkbox state
            # if select_all and not st.session_state.is_select_all:
            #     st.session_state.selected_subdivs = list(all_subdivs)
            #     st.session_state.is_select_all = True
            # elif not select_all and st.session_state.is_select_all:
            #     st.session_state.selected_subdivs = []
            #     st.session_state.is_select_all = False
            #aman end
            # Multiselect for manual subdivision selection
            st.session_state.selected_subdivs = st.multiselect(
                "Choose Subdivisions (one or more):",
                options=all_subdivs,
                default=st.session_state.selected_subdivs,
                key="global_subdiv_select",
                help="Select one or more subdivisions to include in scheduling and visualizations"
            )
            
            # Update is_select_all based on multiselect state
            # ----->>>>>>>*** yaha bhi neeche
            # if len(st.session_state.selected_subdivs) == len(all_subdivs) and not st.session_state.is_select_all:
            #     st.session_state.is_select_all = True
            # elif len(st.session_state.selected_subdivs) != len(all_subdivs) and st.session_state.is_select_all:
            #     st.session_state.is_select_all = False
        
        with st.expander("📁 Data Selection", expanded=True):
            st.subheader("Subdivision Data")
            use_default_data = st.checkbox("Use Default Subdivision Data", value=True, key="data_check")
            if not use_default_data:
                data_file = st.file_uploader("Upload Subdivision Data (CSV)", type="csv", key="data_file")
                if data_file is not None:
                    st.session_state.data = pd.read_csv(data_file)
                    # Reset selections if new data is uploaded
                    st.session_state.selected_subdivs = []
                    # ----->>>>>>>*** yaha bhi neeche
                    # st.session_state.is_select_all = False
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
                step=0.1, 
                key="base_productivity",
                help="Expected meters installed per installer per day under normal conditions"
            )
            
            st.markdown("**Ramp-up Phase**")
            col1, col2 = st.columns(2)
            with col1:
                ramp_up_period = st.slider(
                    "Ramp-up Period (%)", 
                    5, 30, 
                    st.session_state.get('ramp_up_period', 15), 
                    key="ramp_up_period",
                    help="Percentage of timeline dedicated to ramping up installation capacity"
                )
            with col2:
                ramp_up_factor = st.slider(
                    "Ramp-up Factor", 
                    0.1, 1.0, 
                    st.session_state.get('ramp_up_factor', 0.5), 
                    step=0.1, 
                )
            st.markdown("**Ramp-down Phase**")
            col1, col2 = st.columns(2)
            with col1:
                saturation_threshold = st.slider(
                    "Saturation Threshold (%)", 
                    50, 95, 
                    int(st.session_state.get('saturation_threshold', 0.85) * 100), 
                    key="saturation_threshold_slider",
                    help="Percentage completion when ramp-down begins"
                )
            with col2:
                ramp_down_factor = st.slider(
                    "Ramp-down Factor", 
                    0.1, 1.0, 
                    st.session_state.get('ramp_down_factor', 0.7), 
                    step=0.1, 
                    key="ramp_down_factor",
                    help="Productivity during ramp-down phase"
                )
            st.markdown("**Special Conditions**")
            holiday_factor = st.slider(
                "Holiday Productivity Factor", 
                0.0, 1.0, 
                st.session_state.get('holiday_factor', 0.1), 
                step=0.05, 
                key="holiday_factor",
                help="Productivity on holidays as fraction of base productivity"
            )
            slow_period_factor = st.slider(
                "Slow Period Factor", 
                0.1, 1.0, 
                st.session_state.get('slow_period_factor', 0.7), 
                step=0.1, 
                key="slow_period_factor",
                help="Productivity during slow periods as fraction of base productivity"
            )
            is_slow_day = st.selectbox(
                "Apply Slow Day to All Days", 
                ["No", "Yes"], 
                index=0 if st.session_state.get('is_slow_day', 'No') == 'No' else 1, 
                key="is_slow_day",
                help="Apply slow period factor to all days if needed"
            )
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
    st.session_state.selected_subdivs = []  # Empty list, no pre-selection
if 'selected_plots' not in st.session_state:
    st.session_state.selected_plots = [
        "Bell-Shaped Installation Curve",
        "Monthly Meter Installation",
        "Monthly Installer Requirements",
        "Cumulative Progress"
    ]
#here i have hide all substaion function
# ----->>>>>>>*** yaha bhi neeche
# if 'is_select_all' not in st.session_state:
#     st.session_state.is_select_all = False  # Track select-all state
# Initialize scheduling parameters to avoid AttributeError
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
    st.session_state.saturation_threshold = 0.85  # Default value as float
if 'ramp_down_factor' not in st.session_state:
    st.session_state.ramp_down_factor = 0.7
if 'holiday_factor' not in st.session_state:
    st.session_state.holiday_factor = 0.1
if 'slow_period_factor' not in st.session_state:
    st.session_state.slow_period_factor = 0.7
if 'is_slow_day' not in st.session_state:
    st.session_state.is_slow_day = 'No'  # Track select-all state separately


# def render_sidebar():
#     with st.sidebar:
#         st.header("⚙️ Configuration", divider='gray')
        
#         with st.expander("📍 Subdivision Selection", expanded=True):
#             st.subheader("Select Subdivisions")
#             all_subdivs = st.session_state.data['SubDiv'].unique()
#             st.session_state.selected_subdivs = st.multiselect(
#                 "Choose Subdivisions (one or more):",
#                 options=all_subdivs,
#                 default=st.session_state.selected_subdivs,  # Will be [] initially
#                 key="global_subdiv_select",
#                 help="Select one or more subdivisions to include in scheduling and visualizations"
#             )
        
#         # Rest of the sidebar code remains unchanged...
#         with st.expander("📁 Data Selection", expanded=True):
#             st.subheader("Subdivision Data")
#             use_default_data = st.checkbox("Use Default Subdivision Data", value=True, key="data_check")
#             if not use_default_data:
#                 data_file = st.file_uploader("Upload Subdivision Data (CSV)", type="csv", key="data_file")
#                 if data_file is not None:
#                     st.session_state.data = pd.read_csv(data_file)
#             else:
#                 st.session_state.data = data

#             st.subheader("Holidays")
#             use_default_holidays = st.checkbox("Use Default Holidays", value=True, key="holidays_check")
#             if not use_default_holidays:
#                 holidays_file = st.file_uploader("Upload Holidays (CSV)", type="csv", key="holidays_file")
#                 if holidays_file is not None:
#                     st.session_state.all_holidays = pd.read_csv(holidays_file, header=None, names=['holidays'])
#             else:
#                 st.session_state.all_holidays = all_holidays

#             st.subheader("Slow Days")
#             use_default_slow_days = st.checkbox("Use Empty Slow Days", value=True, key="slow_days_check")
#             if not use_default_slow_days:
#                 slow_days_input = st.text_area("Enter slow days (YYYY-MM-DD, one per line)", height=100, key="slow_days_input")
#                 slow_days_list = [date.strip() for date in slow_days_input.split("\n") if date.strip()]
#                 st.session_state.slow_days_df = pd.DataFrame(slow_days_list, columns=['slow_days']) if slow_days_list else pd.DataFrame(columns=['slow_days'])
#             else:
#                 st.session_state.slow_days_df = slow_days_df

#         with st.expander("📅 Scheduling Parameters", expanded=True):
#             st.markdown("**Timeline Settings**")
#             col1, col2 = st.columns(2)
#             with col1:
#                 start_date = st.date_input("Start Date", value=st.session_state.get('start_date', datetime(2024, 8, 1)), key="start_date")
#             with col2:
#                 end_date = st.date_input("End Date", value=st.session_state.get('end_date', datetime(2026, 1, 31)), key="end_date")
            
#             st.markdown("**Productivity Settings**")
#             base_productivity = st.number_input(
#                 "Base Productivity (meters/installer/day)", 
#                 value=st.session_state.get('base_productivity', 7.0), 
#                 min_value=1.0, 
#                 step=0.1, 
#                 key="base_productivity",
#                 help="Expected meters installed per installer per day under normal conditions"
#             )
            
#             st.markdown("**Ramp-up Phase**")
#             col1, col2 = st.columns(2)
#             with col1:
#                 ramp_up_period = st.slider(
#                     "Ramp-up Period (%)", 
#                     5, 30, 
#                     st.session_state.get('ramp_up_period', 15), 
#                     key="ramp_up_period",
#                     help="Percentage of timeline dedicated to ramping up installation capacity"
#                 )
#             with col2:
#                 ramp_up_factor = st.slider(
#                     "Ramp-up Factor", 
#                     0.1, 1.0, 
#                     st.session_state.get('ramp_up_factor', 0.5), 
#                     step=0.1, 
#                     key="ramp_up_factor",
#                     help="Initial productivity as fraction of base productivity"
#                 )
            
#             st.markdown("**Ramp-down Phase**")
#             col1, col2 = st.columns(2)
#             with col1:
#                 saturation_threshold = st.slider(
#                     "Saturation Threshold (%)", 
#                     50, 95, 
#                     int(st.session_state.get('saturation_threshold', 0.85) * 100), 
#                     key="saturation_threshold_slider",
#                     help="Percentage completion when ramp-down begins"
#                 )
#             with col2:
#                 ramp_down_factor = st.slider(
#                     "Ramp-down Factor", 
#                     0.1, 1.0, 
#                     st.session_state.get('ramp_down_factor', 0.7), 
#                     step=0.1, 
#                     key="ramp_down_factor",
#                     help="Productivity during ramp-down phase"
#                 )
            
#             st.markdown("**Special Conditions**")
#             holiday_factor = st.slider(
#                 "Holiday Productivity Factor", 
#                 0.0, 1.0, 
#                 st.session_state.get('holiday_factor', 0.1), 
#                 step=0.05, 
#                 key="holiday_factor",
#                 help="Productivity on holidays as fraction of base productivity"
#             )
#             slow_period_factor = st.slider(
#                 "Slow Period Factor", 
#                 0.1, 1.0, 
#                 st.session_state.get('slow_period_factor', 0.7), 
#                 step=0.1, 
#                 key="slow_period_factor",
#                 help="Productivity during slow periods as fraction of base productivity"
#             )
#             is_slow_day = st.selectbox(
#                 "Apply Slow Day to All Days", 
#                 ["No", "Yes"], 
#                 index=0 if st.session_state.get('is_slow_day', 'No') == 'No' else 1, 
#                 key="is_slow_day",
#                 help="Apply slow period factor to all days if needed"
#             )

#             # Convert saturation threshold back to float (0.5–0.95) for usage
#             st.session_state.saturation_threshold = saturation_threshold / 100.0
# # Initialize session state
# if 'data' not in st.session_state:
#     st.session_state.data = data
# if 'all_holidays' not in st.session_state:
#     st.session_state.all_holidays = all_holidays
# if 'slow_days_df' not in st.session_state:
#     st.session_state.slow_days_df = slow_days_df
# if 'schedule_df' not in st.session_state:
#     st.session_state.schedule_df = None
# if 'selected_subdivs' not in st.session_state:
#     st.session_state.selected_subdivs = []  # Changed to empty list
# if 'selected_plots' not in st.session_state:
#     st.session_state.selected_plots = [
#         "Bell-Shaped Installation Curve",
#         "Monthly Meter Installation",
#         "Monthly Installer Requirements",
#         "Cumulative Progress"
#     ]

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
        using a bell-shaped curve approach that models realistic installation patterns.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("📖 How to Use This Tool", expanded=True):
        st.markdown("""
        ### Step-by-Step Guide
        
        1. **Select Subdivisions**: Choose one or more subdivisions in the sidebar to focus your analysis
        2. **Data Preview**: View and verify your input data (subdivisions, holidays, slow days)
        3. **Schedule Generation**: Configure parameters and generate the installation schedule
        4. **Visual Analysis**: Explore interactive visualizations of the schedule
        
        ### Key Features
        
        - **Bell-shaped installation curve**: Models realistic ramp-up and ramp-down phases
        - **Customizable parameters**: Adjust productivity, phases, and special conditions
        - **Interactive visualizations**: Multiple chart types to analyze the schedule
        - **Export capabilities**: Download schedules and charts for reporting
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Tip**: Use the sidebar to select subdivisions and configure data sources and scheduling parameters.")
    with col2:
        st.success("🚀 **Ready?** Select a page from the sidebar to begin!")



def data_preview_page():
    st.title("🔍 Data Preview")
    st.markdown("""
    Review and verify the input data for the selected subdivisions that will be used for schedule generation.
    """)
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
            st.session_state.data.update(edited_data)
            st.session_state.data = st.session_state.data.copy()
            st.success("Subdivision data updated!")
    
    with st.expander("🎉 Holidays", expanded=True):
        st.markdown("List of dates that will be treated as holidays with reduced productivity.")
        edited_holidays = st.data_editor(st.session_state.all_holidays, num_rows="dynamic", use_container_width=True)
        if st.button("Save Holiday Data Changes"):
            st.session_state.all_holidays = edited_holidays
            st.success("Holiday data updated!")
    
    with st.expander("🐢 Slow Days", expanded=True):
        st.markdown("Dates with reduced productivity (e.g., extreme weather, supply delays).")
        if st.session_state.slow_days_df.empty:
            st.info("No slow days currently defined.")
        else:
            edited_slow_days = st.data_editor(st.session_state.slow_days_df, num_rows="dynamic", use_container_width=True)
            if st.button("Save Slow Days Changes"):
                st.session_state.slow_days_df = edited_slow_days
                st.success("Slow days data updated!")
    
    st.markdown("---")
    st.success("✅ Data verified! Proceed to the Schedule page to generate your installation plan.")

def schedule_page():
    st.title("📅 Generate Schedule")
    st.markdown("""
    Create your meter installation schedule for the selected subdivisions based on the configured parameters.
    The schedule will follow a bell-shaped curve with distinct ramp-up, peak, and ramp-down phases.
    """)
    st.markdown("---")
    
    if not st.session_state.selected_subdivs:
        st.warning("⚠️ Please select at least one subdivision in the sidebar.")
        return
    
    # Display key parameters for review
    with st.expander("🔎 Current Parameters Summary", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Start Date", st.session_state.start_date.strftime('%Y-%m-%d'))
            st.metric("Base Productivity", f"{st.session_state.base_productivity} meters/installer/day")
            st.metric("Ramp-up Period", f"{st.session_state.ramp_up_period}% of timeline")
        with col2:
            st.metric("End Date", st.session_state.end_date.strftime('%Y-%m-%d'))
            st.metric("Saturation Threshold", f"{int(st.session_state.saturation_threshold*100)}%")
            st.metric("Ramp-up Factor", f"{st.session_state.ramp_up_factor*100}% of base")
        with col3:
            st.metric("Total Days", (st.session_state.end_date - st.session_state.start_date).days + 1)
            st.metric("Ramp-down Factor", f"{st.session_state.ramp_down_factor*100}% of base")
            st.metric("Holiday Factor", f"{st.session_state.holiday_factor*100}% of base")
    
    if st.button("✨ Generate Schedule", type="primary", use_container_width=True):
        with st.spinner("Generating schedule... This may take a moment for large projects..."):
            try:
                st.session_state.schedule_df = run_base_mru_schedule(
                    sub_mru_data=st.session_state.data,
                    all_holidays=st.session_state.all_holidays,
                    slow_days_df=st.session_state.slow_days_df,
                    start_date=st.session_state.start_date,
                    end_date=st.session_state.end_date,
                    base_productivity=st.session_state.base_productivity,
                    saturation_threshold=st.session_state.saturation_threshold,
                    ramp_up_period=st.session_state.ramp_up_period,
                    ramp_up_factor=st.session_state.ramp_up_factor,
                    ramp_down_factor=st.session_state.ramp_down_factor,
                    holiday_factor=st.session_state.holiday_factor,
                    slow_period_factor=st.session_state.slow_period_factor,
                    is_slow_day=st.session_state.is_slow_day
                )
                
                st.success("Schedule generated successfully!")
                
                # Show summary statistics
                total_meters = st.session_state.schedule_df.groupby('SubDiv')['Total_Meters'].first().sum()
                installed = st.session_state.schedule_df.groupby('SubDiv')['Cumulative_Meters_Installed'].last().sum()
                completion_percent = (installed / total_meters) * 100
                max_installers = st.session_state.schedule_df['Installers'].max()
                avg_productivity = st.session_state.schedule_df['Productivity'].mean()
                
                col1, col2, col3 = st.columns(3)
                # col1.metric("Total Meters", f"{total_meters:,}")
                # col2.metric("Will Be Installed", f"{installed:,} ({completion_percent:.1f}%)")
                # col3.metric("Peak Installers Needed", max_installers)
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
                
                # Download options
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
                st.error("Please check your input data and parameters and try again.")
    
    elif st.session_state.schedule_df is not None:
        st.subheader("Existing Schedule")
        st.info("A previously generated schedule exists. You can regenerate with new parameters or proceed to visualizations.")
        
        # Show summary of existing schedule
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
    st.markdown("""
    Interactive visualizations of your meter installation schedule for the selected subdivisions.
    Select chart types to explore different aspects of your plan.
    """)
    st.markdown("---")
    
    if st.session_state.schedule_df is None:
        st.warning("⚠️ No schedule found. Please generate a schedule on the Schedule page first.")
        return
    
    if not st.session_state.selected_subdivs:
        st.warning("⚠️ Please select at least one subdivision in the sidebar.")
        return
    
    # Visualization controls (only for plot selection, subdivision selection is now global)
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
    
    # Dynamic visualization display
    for plot in st.session_state.selected_plots:
        with st.container():
            st.subheader(plot)
            
            if plot == "Bell-Shaped Installation Curve":
                st.markdown("""
                **Daily installation progress** showing the bell-shaped curve pattern with:
                - **Solid line**: Meters installed each day
                - **Dashed line**: Installer count (scaled to match meter scale)
                """)
                fig = plot_installation_curve(st.session_state.schedule_df, st.session_state.selected_subdivs)
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot == "Monthly Meter Installation":
                st.markdown("""
                **Monthly totals** showing installation volume by subdivision.
                Use this to identify peak installation months and resource needs.
                """)
                fig = plot_monthly_installation(st.session_state.schedule_df, st.session_state.selected_subdivs)
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot == "Monthly Installer Requirements":
                st.markdown("""
                **Average installers needed per day** by month.
                Helps with workforce planning and resource allocation.
                """)
                fig = plot_installer_requirements(st.session_state.schedule_df, st.session_state.selected_subdivs)
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot == "Cumulative Progress":
                st.markdown("""
                **Progress over time** showing cumulative meters installed versus total target.
                The bar chart shows final progress against targets with completion percentages.
                """)
                figs = plot_cumulative_progress(st.session_state.schedule_df, st.session_state.selected_subdivs)
                for fig in figs:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Download option for each plot
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
                    href = f'<a href="data:image/svg+xml;base64,{b64}" download="{plot.lower().replace(" ", "_")}.svg">⬇️ Download as SVG (vector)</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            st.markdown("---")

# Define pages dictionary
pages = {
    "🏠 Home": home_page,
    "🔍 Data Preview": data_preview_page,
    "📅 Schedule": schedule_page,
    "📈 Visualizations": visualizations_page
}

# Render sidebar
render_sidebar()

# Navigation with icons
st.sidebar.header("🧭 Navigation", divider='gray')
selection = st.sidebar.radio("Go to", list(pages.keys()), label_visibility="collapsed")

# Run selected page
pages[selection]()