import os
import datetime as dt
import glob
import lovely_logger as logging
import streamlit as st
from pathlib import Path
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from orgparse import load

org_time_format = "%Y-%m-%d"


def is_done(item):
    return item.todo == "DONE"


path = Path(__file__)
ROOT_DIR = path.parent.absolute()
home_path = str(Path.home())



@st.cache
def init_logger():
    T = dt.datetime.now()
    logging.info(f"init_logger at {T}")
    name = f"tmp_{T.year}-{T.month:02}-{T.day:02}_{T.hour:02}-{T.minute:02}-{T.second:02}.log"
    logfile = os.path.join(ROOT_DIR, name)
    logging.FILE_FORMAT = "[%(asctime)s] [%(levelname)-8s] - %(message)s"
    logging.DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    logging.init(logfile, to_console=False)

    return logfile


def make_title(durations_seconds):
    mean_d = np.mean(durations_seconds)
    mean_h = seconds_time(mean_d)[0]
    mean_m = seconds_time(mean_d)[1]
    max_d = np.max(durations_seconds)
    min_d = np.min(durations_seconds)
    std_d = np.std(durations_seconds)
    min_h = seconds_time(min_d)[0]
    min_min = seconds_time(min_d)[1]
    max_h = seconds_time(max_d)[0]
    max_min = seconds_time(max_d)[1]
    std_h = seconds_time(std_d)[0]
    std_min = seconds_time(std_d)[1]
    title = f"Mean working time $(h:min)=$ {mean_h:0.0f}:{mean_m:0.0f} $\in$ [{min_h:0.0f}:{min_min:0.0f} - {max_h:0.0f}:{max_min:0.0f}] (±{std_h:0.0f}:{std_min:0.0f})"
    return title


def seconds_time(seconds):
    hours = seconds // 3600
    # hours = seconds / 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return hours, minutes, seconds


def get_start_end_duration_ntasks_day(filename):
    start_times = []
    end_times = []
    day_duration = dt.timedelta(seconds=0)
    root = load(filename)
    logging.info(f"file {filename} has {len(root.children)} items")
    day = filename.split("/")[-1].split(".org")[0]
    for item in root.children:
        logging.info(f"Got item  <{item.heading}>")
        if not is_done(item):
            logging.warning(f"item  <{item.heading}> not done yet!")
            logging.warning(f"\tday  <{day}>")
            continue

        start, end, duration = get_start_end_duration_item(item)
        if isinstance(duration, dt.timedelta):
            start_times.append(start)
            end_times.append(end)
            day_duration += duration

    start_times.sort()
    end_times.sort()
    if start_times and end_times:
        return start_times[0], end_times[-1], day_duration, len(root.children)

    return -1, -1, -1, -1


def get_start_end_duration_item(item):
    clocks = item.clock
    duration = dt.timedelta(seconds=0)
    start_times = []
    end_times = []
    if not clocks:
        logging.warning(f"item <{item.heading}> has no times")
        return -1, -1, -1

    for c in clocks:
        duration += c.duration
        start_times.append(c.start)
        end_times.append(c.end)
        logging.info(f">> <{item.heading}>, {c.start}, {c.duration}")

    start_times.sort()
    end_times.sort()
    return start_times[0], end_times[-1], duration


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_histogram(dates, tasks, nbins):
    df_dates = pd.DataFrame(data = dates)
    df_tasks = pd.DataFrame(data = tasks)
    data = pd.concat([df_dates, df_tasks])
    df = pd.DataFrame(
        {'dates': dates,
         'tasks': tasks,
         }
    )    
    hist = px.histogram(
        df,
        x="dates",
        y="tasks",
        #marginal="rug",
        hover_data=df.columns,
        labels={"waiting": "Waiting time"},
        text_auto=True,
        nbins=nbins,
        #title=f'<b>Maximal waiting time: {maxt:.2f} [s]</b>',
    )
    hist.update_layout(bargap=0.2)
    return hist

@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_durations(dates, durations):
    durations_hour = [seconds_time(duration)[0] for duration in durations]
    fig = make_subplots(
        rows=1,
        cols=1,
        #subplot_titles=[f"<b>{title}</b>"],
        x_title="Day",
        y_title="Duration / h",
    )
    
    trace = go.Scatter(
        x=dates,
        y=durations_hour,
        mode="lines+markers",
        showlegend=False,
        line=dict(width=3),
        marker=dict(size=10),
    )
    fig.append_trace(trace, row=1, col=1)        
    fig.update_layout(hovermode="x")
    return fig


if __name__ == '__main__':
    st.set_page_config(
        page_title="Org-working times",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_logger()
    dfiles = "/Users/chraibi/Dropbox/Orgfiles/org-files/org-roam/daily/*.org"
    files = glob.glob(dfiles)
    files.sort()
    durations = []
    dates = []
    tasks = []
    for filename in files:
        start, end, duration, num_tasks = get_start_end_duration_ntasks_day(filename)
        if isinstance(duration, dt.timedelta):
            durations.append(duration.seconds)
            dates.append(start.date())
            tasks.append(num_tasks)

            
    fig2 = plot_durations(dates, durations)
    t = make_title(durations)
    st.header(t)
    st.plotly_chart(fig2, use_container_width=True)

    # histogram
    nbins = st.slider(
            "Number of bins", 10, 100, value=50, help="Number of bins", key="hist"
        )
    
    fig1 = plot_histogram(dates, tasks, nbins)
    
    st.plotly_chart(fig1, use_container_width=True)

    
    
