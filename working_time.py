import datetime as dt
import glob
import lovely_logger as logging
import streamlit as st

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from orgparse import load

org_time_format = "%Y-%m-%d"


def is_done(item):
    return item.todo == "DONE"


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
    title = f"net time $\mu (h:min):$ {mean_h:0.0f}:{mean_m:0.0f} $\in$ [{min_h:0.0f}:{min_min:0.0f} - {max_h:0.0f}:{max_min:0.0f}] (±{std_h:0.0f}:{std_min:0.0f})"
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
    for item in root.children:
        logging.info(f"Got item  <{item.heading}>")
        if not is_done(item):
            logging.warning(f"item  <{item.heading}> not done yet!")

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
        logging.info(item.heading, c.start, c.duration)

    start_times.sort()
    end_times.sort()
    return start_times[0], end_times[-1], duration


def plot_durations(dates, durations):
    durations_hour = [seconds_time(duration)[0] for duration in durations]
    plt.subplot(311)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(org_time_format))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.plot(dates, durations_hour, "o-", color="gray", lw=0.7)
    plt.ylabel("Duration [h]")
    plt.ylim([0, np.max(durations_hour) + 1])
    plt.grid(alpha=0.2)
    plt.gcf().autofmt_xdate()
    title = make_title(durations)
    plt.title(title)
    plt.show()


def plot_histogram(dates, tasks):
    # Hist number of tasks per day
    ax = plt.subplot(111)
    p1 = ax.bar(dates, tasks, width=0.5, color="gray")
    ax.bar_label(p1, label_type="center")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(org_time_format))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.ylabel("Number of tasks per day")
    plt.gcf().autofmt_xdate()
    plt.grid(alpha=0.2)


if __name__ == '__main__':
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

    plot_histogram(dates, tasks)
    plot_durations(dates, durations)
