import datetime as dt
import glob
import os
from pathlib import Path

import lovely_logger as logging

# import matplotlib.dates as mdates
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from orgparse import load
from plotly.subplots import make_subplots

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
    tags = []
    day_duration = dt.timedelta(seconds=0)
    root = load(filename)
    logging.info(f"file {filename} has {len(root.children)} items")
    day = filename.split("/")[-1].split(".org")[0]
    for item in root.children:
        logging.info(f"Got item  <{item.heading}>")
        if not is_done(item):
            logging.warning(f"item  <{item.heading}> not done yet!")
            logging.warning(f"\tday  <{day}>")
            # children of children
            for c in item.children:
                logging.info(f"Got subitem item  <{c.heading}>")
                if not is_done(c):
                    logging.warning(f"item  <{c.heading}> not done yet!")
                    continue

                start, end, duration, tag = get_start_end_duration_item(c)
                if isinstance(duration, dt.timedelta):
                    # print(f"start {start}, end {end},  duration {duration}")
                    start_times.append(start)
                    end_times.append(end)
                    day_duration += duration
                    tags.append(tag)

        start, end, duration, tag = get_start_end_duration_item(item)
        if isinstance(duration, dt.timedelta):
            start_times.append(start)
            end_times.append(end)
            day_duration += duration
            tags.append(tag)

    start_times.sort()
    end_times.sort()
    if start_times and end_times:
        return start_times[0], end_times[-1], day_duration, len(root.children), tags

    return -1, -1, -1, -1, []


def get_start_end_duration_item(item):
    clocks = item.clock
    duration = dt.timedelta(seconds=0)
    start_times = []
    end_times = []
    tag = item.heading.split(":")[-1]
    logging.info(f"item {item.heading} Tag: <{tag}>")
    if not clocks:
        logging.warning(f"item <{item.heading}> has no times")
        return -1, -1, -1, []

    for c in clocks:
        duration += c.duration
        start_times.append(c.start)
        end_times.append(c.end)
        logging.info(f">> <{item.heading}>, start {c.start}, {c.duration}")

    start_times.sort()
    end_times.sort()
    return start_times[0], end_times[-1], duration, tag


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_histogram(df, nbins):
    hist = px.histogram(
        df,
        x="date",
        y="tasks",
        # marginal="rug",
        hover_data=df.columns,
        labels={"waiting": "Waiting time"},
        text_auto=True,
        nbins=nbins,
        # title=f'<b>Maximal waiting time: {maxt:.2f} [s]</b>',
    )
    hist.update_layout(bargap=0.2)
    return hist


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_durations(dates, durations):
    durations_hour = [seconds_time(duration)[0] for duration in durations]
    fig = make_subplots(
        rows=1,
        cols=1,
        # subplot_titles=[f"<b>{title}</b>"],
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


@st.cache(suppress_st_warning=True, hash_funcs={go.Figure: lambda _: None})
def plot_field(dates, y, ytext):

    fig = make_subplots(
        rows=1,
        cols=1,
        # subplot_titles=[f"<b>{title}</b>"],
        x_title="Day",
        y_title=ytext,
    )

    trace = go.Scatter(
        x=dates,
        y=y,
        mode="lines+markers",
        showlegend=False,
        line=dict(width=3),
        marker=dict(size=10),
    )
    fig.append_trace(trace, row=1, col=1)
    fig.update_layout(hovermode="x")
    return fig


if __name__ == "__main__":
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
    start_times = []
    end_times = []
    years = []
    months = []
    days = []
    tags = []
    # genre = st.radio("Aggregate", ("Daily", "Monthly", "Yearly"))
    files_df = []
    for filename in files:
        date = filename.split("/")[-1].split(".org")[0]
        month = date.split("-")[1]
        year = date.split("-")[0]
        day = date.split("-")[2]

        (
            start,
            end,
            duration,
            num_tasks,
            tag_per_day,
        ) = get_start_end_duration_ntasks_day(filename)
        if isinstance(duration, dt.timedelta):
            days.append(day)
            years.append(year)
            months.append(month)
            durations.append(duration.seconds)
            dates.append(start.date())
            tasks.append(num_tasks)
            end_times.append(end.hour)
            start_times.append(start.hour)
            tags.append(tag_per_day)
            files_df.append(filename.split("/")[-1])
        else:
            logging.warning(f"hmm {filename}")
    df = pd.DataFrame(
        {
            "files": files_df,
            "date": dates,
            "year": years,
            "month": months,
            "day": days,
            "start": start_times,
            "end": end_times,
            "duration": [d // 3600 for d in durations],
            "tasks": tasks,
            # "tags": tags
        }
    )
    fig2 = plot_durations(dates, durations)
    t = make_title(durations)
    st.header(t)
    st.plotly_chart(fig2, use_container_width=True)

    # histogram
    st.header("Tasks per day")
    nbins = st.slider(
        "Number of bins", 10, 100, value=90, help="Number of bins", key="hist"
    )

    fig1 = plot_histogram(df, nbins)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("--------")
    c1, c2 = st.columns((1, 1))

    monthly = df.groupby(["month", "year"])
    months = monthly.groups.keys()
    # months.extend(('04', '2022'))

    choose_month = c1.selectbox("Month", months)
    choose_field = c2.selectbox("Field", ("duration", "tasks", "start", "end"))
    d_mean, d_std, d_sum, d_min, d_max = monthly.get_group(choose_month)[
        choose_field
    ].agg([np.mean, np.std, np.sum, np.min, np.max])

    fig = plot_field(
        monthly.get_group(choose_month)["date"],
        monthly.get_group(choose_month)[choose_field],
        choose_field,
    )
    c2.plotly_chart(fig, use_container_width=True)
    fig = plot_field(
        np.unique(df["month"]), df.groupby("month")[choose_field].mean(), choose_field
    )
    c1.plotly_chart(fig, use_container_width=True)

    fig = plot_field(
        np.unique(df["month"]),
        df.groupby("month")[choose_field].count(),
        "working days",
    )
    c1.plotly_chart(fig, use_container_width=True)

    count = monthly.get_group(choose_month)[choose_field].count()
    c2.write(f"## Date {choose_month[0]}/{choose_month[1]}")
    c2.write(f"#### {choose_field}")
    c2.write(
        f"""
    - working days: {count}
    - sum: {d_sum}
    - min: {d_min}
    - max: {d_max}
    - mean: {d_mean:.2f}   ($\pm$ {d_std:.2f})
    """
    )

    # c1.code(monthly.get_group(choose_month)[choose_field] )

    st.header("Summary")
    st.dataframe(df)

    # def file_selector(folder_path='.'):
    #     filenames = os.listdir(folder_path)
    #     selected_filename = st.selectbox('Select a file', filenames)
    #     return os.path.join(folder_path, selected_filename)

    # user_input = st.text_input("Paste the directory of your audio file")
    # if len(user_input) != 0:
    #     src = file_selector(folder_path=user_input)
    #     st.code(src)
