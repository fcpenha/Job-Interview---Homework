# For running this code, you may use:
# ipython -i script.py

# %%%%%%%%%%%%%%%%
# %%% PREAMBLE %%%
# %%%%%%%%%%%%%%%%

# Common Python libraries
import numpy as np
from matplotlib import pyplot as plt

# Data related Python libraries
import json
import datetime

# Non-Linear Least-Square Minimization and Curve-Fitting
import lmfit

# Pyplot configuration
fs = 24
plt.rc('text', usetex=True, fontsize=fs)
plt.rc('ytick', labelsize=fs)
plt.rc('xtick', labelsize=fs)
plt.rc('axes', labelsize=fs)
plt.rc('legend', fontsize=fs - 6)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

# %%%%%%%%%%%%%
# %%% BEGIN %%%
# %%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%
# %%% READ DATA %%%
# %%%%%%%%%%%%%%%%%

# Read data from "sample.txt"
data = json.load(open('./sample.txt'))

# number of entries
n_entries = len(data)

lst = {}

# list total
lst['total'] = np.asarray([
                              data[k]['complemento']['valorTotal']
                              for k in range(0, n_entries)])

# There seems to be no relevant field, for forecasting,
# profit, except for the date and time of the transactions.
# I assume that the day of the week is relevant, because
# of consumers' behaviour. Their preferences may depend
# on many things, for example, in a business area
# restaurants may be packed for lunch on Mondays,
# while the same logic does not apply for a
# restaurant in a bohemian area. This way, each
# restaurant may present unique trends. We choose to
# focus on the trend presented in the sample document.


# list weekdays (Monday=1, Tuesday=2, and so on)
lst['weekdays'] = np.asarray([
                                 datetime.datetime.strptime(data[k]['ide']['dhEmi']['$date'],
                                                            '%Y-%m-%dT%H:%M:%S.%fZ').isoweekday()
                                 for k in range(0, n_entries)])

# The file contains 2 weeks: It begins on a Tuesday and ends on a Saturday.
#
# In [9]: lst['weekdays'][0]
# Out[9]: 2
#
# In [11]: lst['weekdays'][-1]
# Out[11]: 6

isoweek = np.asarray(range(1, 8))

weekdays_mask = {}

for w in isoweek:
    weekdays_mask[w] = lst['weekdays'] == w

# List totals for each weekday (multiply totals with mask and then sum everything)
# in units of thousands
for w in isoweek:
    lst['total', w] = np.sum(weekdays_mask[w] * lst['total']) / 1000.

# List mean based on 2 week data
# in units of thousands
for w in isoweek:
    lst['mean', w] = lst['total', w] / 2.


# %%%%%%%%%%%
# %%% FIT %%%
# %%%%%%%%%%%

# Model: 5th order polynomial for 7 weekdays
# Model must be at most 1 order lower than number of data points
# For example, a straight line (order 1) can fit two points or more


def polyn(t, c0, c1, c2, c3, c4, c5):
    return c0 + c1 * t + c2 * t ** 2 + c3 * t ** 3 + c4 * t ** 4 + c5 * t ** 5


model = lmfit.Model(polyn, independent_vars=['t'])

cmin = -100.
cmax = +100.

fit_result = model.fit(
    [lst['mean', w] for w in isoweek], t=isoweek,
    c0=lmfit.Parameter(value=0., min=cmin, max=cmax),
    c1=lmfit.Parameter(value=0., min=cmin, max=cmax),
    c2=lmfit.Parameter(value=0., min=cmin, max=cmax),
    c3=lmfit.Parameter(value=0., min=cmin, max=cmax),
    c4=lmfit.Parameter(value=0., min=cmin, max=cmax),
    c5=lmfit.Parameter(value=0., min=cmin, max=cmax)
)

# 1-sigma limits in "lmfit" may be obtained with
# best_fit = fit_result.params

# Full report in "lmfit" may be obtained with
# print fit_result.fit_report()

# Covariance matrix
cov = fit_result.covar

# %%%%%%%%%%%%%%%%%%%%%%
# %%% COMPUTE ERRORS %%%
# %%%%%%%%%%%%%%%%%%%%%%

# Do the error propagation with the full covariance matrix:


def error(t):
    return np.sum(np.asarray([[
                                  cov[i, j] * (t ** i) * (t ** j)
                                  for i in range(0, 6)]
                              for j in range(0, 6)]))


# Vectorize function
vec_error = np.vectorize(error)

# %%%%%%%%%%%%
# %%% PLOT %%%
# %%%%%%%%%%%%

plt.scatter(isoweek,
            [lst['mean', w] for w in isoweek],
            marker='.', s=300)

# Best-fit
plt.plot(np.linspace(1., 7., num=50),
         polyn(t=np.linspace(1., 7., num=50), **fit_result.values),
         color='red', linewidth=3, label=r'Best-fit')

# Upper 1-sigma limit
plt.plot(np.linspace(1., 7., num=50),
         polyn(t=np.linspace(1., 7., num=50), **fit_result.values) +
         vec_error(t=np.linspace(1., 7., num=50)),
         color='darkorange', linewidth=3)

# ,ower 1-sigma limit
plt.plot(np.linspace(1., 7., num=50),
         polyn(t=np.linspace(1., 7., num=50), **fit_result.values) -
         vec_error(t=np.linspace(1., 7., num=50)),
         color='darkorange', linewidth=3)

# 68.3 confidence band
plt.fill_between(np.linspace(1., 7., num=50),
                 polyn(t=np.linspace(1., 7., num=50), **fit_result.values) +
                 vec_error(t=np.linspace(1., 7., num=50)),
                 polyn(t=np.linspace(1., 7., num=50), **fit_result.values) -
                 vec_error(t=np.linspace(1., 7., num=50)),
                 facecolor='orange', alpha=0.5, linewidth=0.0, label=r'68.3\% confidence')

plt.xlim([1, 7])
plt.ylim([0, 20])

plt.xlabel(r'Iso Week Date')
plt.ylabel(r'(2-week Mean) / 1000')

# Show also weekday names
plt.xticks([1, 2, 3, 4, 5, 6, 7],
           ['1\n Mon', '2\n Tue', '3\n Wed', '4\n Thu', '5\n Fri', '6\n Sat', '7\n Sun'])

plt.legend(loc='upper right', frameon=True, numpoints=1, ncol=1)

plt.tight_layout()
plt.savefig('./plot_analysis.pdf')
plt.close()
