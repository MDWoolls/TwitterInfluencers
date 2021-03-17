import NetworkClass as nc
import pickle
import traceback
import DataAnalysis as da

import matplotlib.pyplot as plt
import numpy as np
import math
from statistics import mean
from collections import defaultdict
import collections
import requests
import time
import heapq

NA = 5
names = ['MDWoolls', 'mikeroweworks', 'CGPGrey', 'notch', 'BradyHaran']
for screen_name in names:
    print(screen_name)
    save_file = f'data/{screen_name}_network.txt'
    for i in range(NA):
        try:
            print(f"{screen_name} - loop {i}")
            _ = nc.twitter_percolation(screen_name, NA=1, watchQ=True, N_stop=250, save_file=save_file)
        except:
            traceback.print_exc()
            print('Saving before quiting')
            with open(save_file, 'wb') as fp:
                pickle.dump(nc.FULL_DATA, fp)
            # raise KeyboardInterrupt

screen_names = names

power_list = []
for name in screen_names:
    power_list.append(da.generate_plot(name))

print(power_list)

output_file('plots/followers_power.html')

power_dict = ColumnDataSource(data={'count':[t[0] for t in power_list],
             'power': [t[1] for t in power_list],
             'error': [t[2] for t in power_list],
             'names': [t[3] for t in power_list]
                                   })

lower = []
upper = []
base = []
for val in power_list:
    lower.append(val[1]-val[2])
    upper.append(val[1]+val[2])
    base.append(val[0])

errors = ColumnDataSource(data ={
    'lower': lower,
    'upper': upper,
    'base': base
})


# file to save the model
output_file(f'plots/followers_power.html')

TOOLS = "hover,pan,wheel_zoom,box_zoom,reset,save"
# instantiating the figure object
graph = figure(title = f'Follower count compared to Influence power',
            x_axis_label = 'Follower count',
            y_axis_label = 'Influence power',
            x_axis_type = 'log',
            tools = TOOLS,
            y_range = find_y_range(power_dict.data),
            plot_width = graph_width)

# graph.scatter('count','power',source=power_dict)
graph.hover.tooltips = [
        ('user name',"@names"),
        ('follower count',"@count"),
        ('infuencer power',"@power \u00B1 @error")
    ]

graph.add_layout(
    Whisker(source = errors, base = 'base', lower = 'lower', upper = 'upper', dimension = 'height')
)

#labels = LabelSet(x='count',y='power',text='names', source=power_dict,level='glyph',render_mode='canvas')
graph.scatter('count','power',source=power_dict)

#graph.add_layout(labels)
show(graph)
save(graph)

# with open('data_dump.txt', 'rb') as fp:
#    data = pickle.load(fp)
'''
n_a = 100

data.sort()
print(data)

data_average = []
i = 0
while i + n_a <= len(data):
    p_val = (data[i + n_a - 1][0] + data[i][0]) / 2
    avg = mean(x[1] for x in data[i:i + n_a])
    data_average.append((p_val, avg))
    i += n_a


x = [d[0] for d in data_average]
y = [d[1] for d in data_average]
p_max = max(x)
i_max = x.index(p_max)

x = x[:i_max+1]
y = y[:i_max+1]

# plt.plot(p_list, sizes, '.')
plt.plot(x, y, '.')
plt.title(f'Growth of {screen_name}\'s network')
plt.xlabel('re-tweet rate')
plt.ylabel('followers count')

plt.savefig(f'{screen_name}_network.pdf', orientation='landscape', format='pdf')
plt.show()
'''
