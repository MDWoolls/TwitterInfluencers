import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show, output_notebook, save
from bokeh.models import Label, LabelSet, ColumnDataSource, Range1d, Whisker
from collections import defaultdict
from scipy.optimize import curve_fit
import math
from requests_oauthlib import OAuth1
import requests
import simplejson as json

output_notebook()
graph_width = 1200


# save_file = f'data/{screen_name}_network.txt'

# file to save the model
# output_file(f'plots/{screen_name}_network_bokeh.html')

def average_data(raw_data, n_a):
    raw_data.sort()

    # data_average = {'p':[],'p_min':[],'p_max':[],'size':[],'size_error':[]}
    data_average = defaultdict(lambda: [])
    i = 0
    while i + n_a <= len(raw_data):
        p_min = raw_data[i][0]
        p_max = raw_data[i + n_a - 1][0]
        data_average['p_min'].append(p_min)
        data_average['p_max'].append(p_max)
        data_average['p'].append((p_min + p_max) / 2)
        y_vals = [x[1] for x in raw_data[i:i + n_a]]
        data_average['size'].append(np.mean(y_vals))
        data_average['size_error'].append(np.std(y_vals) / np.sqrt(len(y_vals)))
        data_average['size_min'].append(min(y_vals))
        data_average['size_max'].append(max(y_vals))
        i += n_a

    if i + 1 < len(raw_data):
        p_min = raw_data[i][0]
        p_max = raw_data[-1][0]
        data_average['p_min'].append(p_min)
        data_average['p_max'].append(p_max)
        data_average['p'].append((p_min + p_max) / 2)
        y_vals = [x[1] for x in raw_data[i:]]
        data_average['size'].append(np.mean(y_vals))
        data_average['size_error'].append(np.std(y_vals) / np.sqrt(len(y_vals)))
        data_average['size_min'].append(min(y_vals))
        data_average['size_max'].append(max(y_vals))

    return pd.DataFrame(data_average, dtype=float)


def fitting_function(x, pc, A, d, C):
    return C + A / (pc - x) ** d


# takes in the already averaged data and finds the first point with positive curvature
def min_cut(data):
    for i in range(1, len(data)):
        ld = (data['size'][i] - data['size'][i - 1]) / (data['p'][i] - data['p'][i - 1])
        rd = (data['size'][i + 1] - data['size'][i]) / (data['p'][i + 1] - data['p'][i])
        if rd >= ld:
            return i
    return 0


def first_sig(n):
    return int(math.log10(n))


with open("twitter_secrets.json.nogit") as fh:
    secrets = json.loads(fh.read())

# create an auth object
auth = OAuth1(
    secrets["api_key"],
    secrets["api_secret"],
    secrets["access_token"],
    secrets["access_token_secret"]
)


def get_id(screen_name):
    user_id = requests.get(
        'https://api.twitter.com/2/users/by/username/' + screen_name,
        auth=auth
    )
    while user_id.status_code == 429:
        time.sleep(60)
        user_id = requests.get(
            'https://api.twitter.com/2/users/by/username/' + screen_name,
            auth=auth
        )
    return user_id.json()['data']['id']


def get_info(screen_name):
    user_id = get_id(screen_name)
    response = requests.get(
        'https://api.twitter.com/2/users/' + user_id,
        auth=auth,
        params={'user.fields': 'location', 'user.fields': 'public_metrics'}
    )
    return response.json()['data']['public_metrics']


def generate_plot(screen_name, save_file=None, out_file=None, show_graph=True, save_graph=True):
    if save_file is None:
        save_file = f'data/{screen_name}_network.txt'

    # file to save the model
    if out_file is None:
        output_file(f'plots/{screen_name}_network_bokeh.html')
    else:
        output_file(out_file)

    with open(save_file, 'rb') as fp:
        raw_data = pickle.load(fp)

    n_a = len(raw_data) // 120
    data = average_data(raw_data, n_a)

    i_max = data['size'].idxmax()
    row_max = data.loc[i_max]

    size_max = row_max['size']
    p_max = row_max['p_max']

    data = data.loc[:i_max]

    i_cut = min_cut(data)
    if i_cut >= i_max:
        i_cut = 0

    fit_data = data.loc[i_cut:]

    pars, cov = curve_fit(f=fitting_function,
                          xdata=fit_data['p'],
                          ydata=fit_data['size'],
                          p0=[p_max, 1, 1, 0],
                          bounds=(0, np.inf))

    p_c = pars[0]
    dp_c = np.sqrt(np.diag(cov))[0]

    n_round = -first_sig(dp_c) + 1
    p_c_string = f'{round(p_c, n_round)} \u00B1 {round(dp_c, n_round)}'

    # plotting the line graph
    # instantiating the figure object
    graph = figure(title=f'Growth of {screen_name}\'s network',
                   x_axis_label='re-tweet rate',
                   y_axis_label='share count',
                   y_axis_type='log',
                   plot_width=graph_width,
                   x_range=[0, p_c + dp_c + 10 ** (-n_round - 1)],
                   y_range=[1, max(1.5 * size_max, fitting_function(p_c - dp_c / 2, *pars))]

                   )

    graph.line(x=data['p'], y=data['size'], legend_label=f'{screen_name}: {p_c_string}')

    x_fit = np.linspace(0, p_c, 1000, endpoint=False)
    graph.line(x=x_fit, y=fitting_function(x_fit, *pars), line_color='green', line_dash='dashed')

    graph.scatter(x=data['p'], y=data['size'], legend_label=f'{screen_name}: {p_c_string}')

    graph.line([p_c, p_c], [1, fitting_function(p_c - 0.00001, *pars)], line_dash='dashed', line_color='red')
    graph.line([p_c - dp_c, p_c - dp_c], [1, fitting_function(p_c - 0.00001, *pars)], line_dash='dashed',
               line_color='blue')
    graph.line([p_c + dp_c, p_c + dp_c], [1, fitting_function(p_c - 0.00001, *pars)], line_dash='dashed',
               line_color='blue')

    graph.legend.location = 'top_left'
    # displaying the model
    if show_graph:
        show(graph)
    if save_graph:
        save(graph)

    return (get_info(screen_name)['followers_count'], p_c, dp_c, screen_name)


# given dict returns min and max
def find_y_range(data):
    mx = 0
    mn = np.inf
    for power, error in zip(data['power'], data['error']):
        lower = power - error * 1.5
        upper = power + error * 1.5
        if upper > mx:
            mx = upper
        if lower < mn:
            mn = lower
    return (mn, mx)
