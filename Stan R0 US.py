# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Fitting $R_t$ From COVID-19 Infection Data
# %% [markdown]
# This notebook was originally based off the `Realtime R0.ipynb` calculation, but intended to produce a Bayesian posterior over the values of $R_t$ *and* $\sigma$ given the data instead of only fitting $R_t$ using a max-likelihood estimate for $\sigma$.  It has now diverged significantly; the underlying model is fairly similar to `Realtime Rt mcmc.ipynb`, but our data handling differs in that we do not attempt to account for delays between infection, onset of symptoms, and confirmation.  Instead, we are estimating $R_t$ *of confirmation* (roughly, the delay time from infection to confirmation is one to two weeks---so our estimate will lag the infection $R_t$ by a week or two).  
# 
# We use Stan to sample over the AR(1) prior for $R_t$ (increments are independent Gaussian variables) and also $\sigma$, the s.d. of the increment distribution. 
# 
# The [Stan](http://mc-stan.org) model we use for the sampling can be found in the `R0.stan` file in the current directory.
# 
# Note that our Python environment is captured in `environment.yml`; issuing
# ```zsh
# conda env create -f environment.yml
# ```
# should duplicate it on your system.  You can then activate it via 
# ```zsh
# conda activate covid-19
# ```

# %%
get_ipython().run_line_magic('pylab', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# %%
import arviz as az
import datetime
import matplotlib.dates as md
import os
import os.path
import pandas as pd
import pystan
import scipy.stats as ss
import seaborn as sns
from tqdm import tqdm

sns.set_context('notebook')
sns.set_palette('colorblind')

# %% [markdown]
# Download the dataset from https://covidtracking.com

# %%
url='https://covidtracking.com/api/v1/states/daily.csv'
states = pd.read_csv(url,
                     usecols=['date', 'state', 'positive', 'negative'],
                     parse_dates=['date'],
                     index_col=['state', 'date'],
                     squeeze=True).sort_index()
states['total'] = states['positive'] + states['negative']

# %% [markdown]
# Compile the Stan model that simultaneously fits for $R_t$ and $\sigma$.  Our model differs from that in http://rt.live in several respects.  Trivially, we choose different priors on $\sigma$ (the scatter from one day to the next in the imposed prior on $R_t$), the first day's infection count, and $R_1$.  
# 
# We differ from the default implementation at http://rt.live more significantly in that we use the last day's *expected* number of counts to predict the current day's expected numbers via $R_t$:
# $$
# \Lambda_{i+1} = \Lambda_i \exp\left( \frac{R_{i+1} - 1}{\tau} \right);
# $$
# we use a Poisson measurement model, where 
# $$
# k_{i+1} \sim \mathrm{Poisson}\left( \Lambda_{i+1} \right);
# $$
# we have tried Binomial measurement models (see `Stan R0-Binomial.ipynb`) and also negative binomial (over-dispersed Poisson), but neither seems to work very well.  The Binomial model suffers because the total numbers of tests are not well curated; also, theoretically, Binomial would only be appropriate if the number of tests on a given day were fixed in advance, which is not very representative of how the testing is done.  Negative binomial is [sometimes used](https://mrc-ide.github.io/covid19estimates/#/) to inflate the uncertainty from a Poisson measurement model; but here, since the $R_t$ timeseries *can* fit any data to arbitrary accuracy, there is no "traction" on $\phi$ (the over-dispersion parameter), and the data prefer pure Poisson models.  We could *impose* a $\phi$ (i.e. over-disperse the Poisson by hand), but I am currently not sure which $\phi$ it would be appropirate to choose.
# 
# http://rt.live instead uses the last day's *observed* counts to predict the current day's expected counts:
# $$
# \Lambda_{i+1} = k_i \exp\left( \frac{R_{i+1}-1}{\tau} \right);
# $$
# they also use a Poisson measurement model:
# $$
# k_{i+1} \sim \mathrm{Poisson}\left( \Lambda_{i+1} \right)
# $$
# The http://rt.live model is odd, in that the *observed* counts are not necessarily the thing that grows exponentially in the SIR model---the observed counts have Poisson sampling noise in them, while the thing that grows exponentially is the *true* number of infected in the population.  The effect of using $k_i$ instead of $\Lambda_i$ is a reduction in the uncertainty on $R_t$, since $k_i$ is conditioned on, and therefore fixed, while $\Lambda_i$ carries some uncertainty.
# 
# One final complication is that we marginalize over the serial number, using a log-normal distribution that matches to a measured mean and s.d.  

# %%
model = pystan.StanModel(file='R0.stan')

# %% [markdown]
# We marginalize over the serial time, based on the mean and standard deviation reported at https://epiforecasts.io/covid/ (which, in turn, come from [Nishiura, Linton, & Akhmetzhanov (2020)](https://doi.org/10.1101/2020.02.03.20019497)).

# %%
serial_mean = 4.7
serial_std = 2.9

# %% [markdown]
# Smoothing has been a constant issue; there are weird day-by-day trends in the data.  For example, here is the time-series of log-odds for NY:

# %%
ny = states.loc['NY']

(log(ny['positive'].diff().dropna() + 1) - log(ny['total'].diff().dropna() - ny['positive'].diff().dropna() + 1)).plot()

# %% [markdown]
# There is some total junk (low numbers!) in the early data, followed by a pretty reasonable rising and then falling trend (yeah!).  But even in the later data, there are some individual day outliers.  In any case, the the distribution of time between infection to symptoms to confirmation via test is quite broad (at least a week or two, with s.d. maybe one week) that it seems reasonable to smooth the time-series some.  In the past I had used an exponential filter, but that does not provide very much smoothing of the high-frequency components unless the timescale is very long; a Gaussian filter is much smoother.  Convolving in the time-domain with a Gaussian 
# $$
# g(\tau) = \exp\left[ -\frac{\tau^2}{2 \sigma^2} \right]
# $$
# is equivalent to multiplication in the frequency domain by 
# $$
# \tilde{g}(f) \propto \exp\left[ -\frac{f^2}{2 \kappa^2} \right],
# $$
# with width 
# $$
# \kappa = \frac{1}{2\pi \sigma}.
# $$
# If we want to suppress fluctuations on longer than 7-day timescales by a factor of 10 (this may be too much smoothing, but let's go with it for now), then we want to choose 
# $$
# \frac{\left( 1 / 7 \, \mathrm{d} \right)^2}{2 \kappa^2} = \log 10,
# $$
# or 
# $$
# \sigma \simeq 2.4 \, \mathrm{d}.
# $$
# 
# Lately http://rt.live has been using a method for projecting cases back to time of onset that effectively filters on a ~7 day timescale; a discussion, including my rationale for not using that method, can be found [here](https://github.com/k-sys/covid-19/issues/43).  Note that smoothing an exponential timeseries with another exponential *does not change the series timeconstant*, so we might think (hope?) that we are still estimating the local $R_t$ in this manner.
# 
# Note that the Gaussian filter is *acausal*; the comment [here](https://github.com/k-sys/covid-19/issues/30), argues for using a causal filter to preserve the "accumulative" property of Bayseian inference.  However, note that our prior on $R_t$ is *already* acausal because we fit $\sigma$ from the full data set.  

# %%
smooth_std = 2.4

# %% [markdown]
# Our model is that $\log R_t$ random-walks with some scale $\sigma$.  Totally off the top of my head, I would not expect $R_t$ to change by a factor larger than, say, $\sqrt{2}$ in one week (so $\log \sigma$ increments by $\log \sqrt{2}$).  Since in a random walk variances add, and $\sigma$ is the scale of the walk step *per day*, we have 
# $$
# 7 \times \sigma^2 \lesssim \left(\log \sqrt{2}\right)^2
# $$
# or 
# $$
# \sigma \lesssim \frac{\log \sqrt{2}}{\sqrt{7}} \simeq 0.13
# $$
# 
# We choose a prior for $\sigma$ that is 
# $$
# \sigma \sim N\left( 0, 0.05 \right),
# $$
# so that this "extreme" value of $\sigma$ lies a bit outside 2-$\sigma$.

# %%
sigma_scale = 0.05

# %% [markdown]
# ## One State By Hand
# %% [markdown]
# Here we do my home state (NY) by hand, just to show aspects of the data processing, fitting, and plotting; stay tuned for the bulk run.  Note that we cut out samples from a time before the state has tested (cumulatively) 1000 people, since these are the "junk" above.

# %%
ny = states.loc['NY']
istart = np.where(ny['total'] > 1000)[0][0]
ny = ny.iloc[istart:]
pos = ny['positive'].diff().dropna().rolling(int(round(5*smooth_std)), min_periods=1, center=True, win_type='gaussian').mean(std=smooth_std)

pos.plot()

data = {
    'ndays': len(pos),
    'k': pos.round().astype(np.int),
    
    'tau_mean': serial_mean,
    'tau_std': serial_std,
    
    'sigma_scale': sigma_scale
}

yscale('log')

# %% [markdown]
# Run the fits, hinting to `arviz` that it should use the appropriate date-time coordinates for the expected number of counts and $R_t$.

# %%
fit_stan = model.sampling(data=data)
fit = az.from_pystan(fit_stan, 
                     coords={'dates': data['k'].index,
                             'Rt_dates': data['k'].index[1:]},
                     dims={'Rt': ['Rt_dates'],
                           'log_odds': ['dates']})

# %%
# temporarily supress UserWarning
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# %% [markdown]
# A quick traceplot, just because it looks pretty:

# %%
az.plot_trace(fit, compact=True)

# %%
# reactivate UserWarnings
warnings.filterwarnings("default", category=UserWarning)


# %%
# customized layout

from datetime import datetime as dtime
from datetime import timedelta

date_now = dtime.now()
date_then = dtime.now() - timedelta(days=48)

rtlive = ["#5db044", "#da5d5d", "#95a5a6", "#e74c3c", "#34495e"]
sns.set_palette(rtlive)

# %%
m = median(fit.posterior.Rt, axis=(0,1))
hh = percentile(fit.posterior.Rt, 97.5, axis=(0,1))
h = percentile(fit.posterior.Rt, 84, axis=(0,1))
l = percentile(fit.posterior.Rt, 16, axis=(0,1))
ll = percentile(fit.posterior.Rt, 2.5, axis=(0,1))

x = fit.posterior.Rt_dates.values
line_1 = np.ones(len(x))

# NOTE: try this approach instead (filling with NaN):
m_good = m.copy()
m_bad = m.copy()
m_good[m_good > 1.30] = np.nan
m_bad[m_bad <= 0.98] = np.nan

fig, ax = plt.subplots()
ax.plot(x, m_good, ls='-', marker='', lw=1.5, color=sns.color_palette()[0])
ax.plot(x, m_bad, ls='-', marker='', lw=1.5, color=sns.color_palette()[1])
    
good_dates = m <= 1.0
bad_dates = m > 1.0

#ax.plot_date(x[good_dates], m[good_dates], ls='-', marker='', xdate=True, color=sns.color_palette()[0])
#ax.fill_between(x, where(h<1, h, 1) , where(l<1, l, 1), alpha=0.10, color=sns.color_palette()[0])

ax.fill_between(x, where(hh<1, hh, 1) , where(ll<1, ll, 1), alpha=0.10, color=sns.color_palette()[0])

#ax.plot_date(x[bad_dates], m[bad_dates], ls='-', marker='', xdate=True, color=sns.color_palette()[1])
#ax.fill_between(x, where(h>1, h, 1) , where(l>1, l, 1), alpha=0.10, color=sns.color_palette()[1])

ax.fill_between(x, where(hh>1, hh, 1) , where(ll>1, ll, 1), alpha=0.10, color=sns.color_palette()[1])

#ax.plot_date(x, m, ls='-', marker='', xdate=True, color=sns.color_palette()[0])
ax.plot(x, line_1, ls='dotted', color = sns.xkcd_rgb["light grey"], lw=3)

ax.set_title('Sweden')

plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.set_ylabel(r'$R_t$')

ax.set_xlim([date_then, date_now])

ax.set_ylim(bottom = 0,
                top = 3,
                auto = False)
sns.despine()
# %% [markdown]
# Something interesting is happening here: we seem to be getting better inferences for $\tau$ (the serial time) than the prior.  This is, in principle, possible---for example, we should get very good inferences on $\tau$ if the positive rate declines significantly, since $R_t \geq 0$ implies that the most extreme decay is $\Lambda(t) \sim \exp\left( -t / \tau \right)$.  Posterior in blue, prior in black (recall that the prior is chosen so that it has the same mean and s.d. as the quoted interval for $\tau$ from previous research).
# 
# Before we make too much of this, we should probably check the effect of our smoothing, which suppresses changes on short timescales (we chose the smoothing time so that Fourier modes with frequencies larger than $1/\left( 7 \, \mathrm{d} \right)$ will be suppressed by an order of magnitude or more); this could be leading to the larger inference for $\tau$.  
# 
# Worth following up on later.

# %%
# reset to orig palette
sns.set_palette('colorblind')


# %%
az.plot_posterior(fit, var_names='tau', )
xs = linspace(2, 12, 1024)
plot(xs, ss.lognorm(0.57, scale=exp(1.4)).pdf(xs), '-k')

# %% [markdown]
# ## All States

# %%
def fit_state(states, state_key):
    st = states.loc[state_key]
    
    istart = np.where(st['total'] > 1000)[0][0]
    st = st.iloc[istart:]

    pos = st['positive'].diff().dropna().rolling(int(round(5*smooth_std)), min_periods=1, center=True, win_type='gaussian').mean(std=smooth_std)
    # tot = st['total'].diff().dropna().rolling(int(round(5*smooth_std)), min_periods=1, center=True, win_type='gaussian').mean(std=smooth_std)
    
    data = {
        'ndays': len(pos),
        'k': pos.round().astype(np.int),
        # 'n': tot.round().astype(np.int),
        
        'tau_mean': serial_mean,
        'tau_std': serial_std,
        'sigma_scale': sigma_scale
    }
    
    if len(pos) == 0:
        raise RuntimeError("can't fit an empty data set")
    
    tries = 0
    iter = 2000
    thin = 1
    while tries < 3:
        fit = az.from_pystan(model.sampling(data=data, iter=iter, thin=thin), 
                             coords={'dates': pos.index,
                                     'Rt_dates': pos.index[1:]},
                             dims={'Rt': ['Rt_dates'],
                                   'log_odds': ['dates']})
        ess = az.ess(fit, var_names=['tau', 'sigma', 'L0', 'Rt']).min()
        if ess.tau > 1e2 and ess.sigma > 1e2 and ess.L0 > 1e2 and ess.Rt > 1e2 and np.all(az.bfmi(fit) > 0.2):
            break
        else:
            tries += 1
            iter *= 2
            thin *= 2
    
    return fit

# %% [markdown]
# We run the fit on each state (takes ten-twenty minutes on my MacBook); some states have no data, so we skip those.

# %%
state_fits = {}
with tqdm(states.groupby(level=0)) as bar:
    for st, _ in bar.iterable:
        try:
            if np.all(np.isnan(states.loc[st]) | (states.loc[st] == 0)):
                bar.write('Skipping state: {:s}'.format(st))
                bar.update(1)
                continue
            state_fits[st] = fit_state(states, st)
            bar.update(1)
        except (RuntimeError, IndexError):
            bar.write('State {:s} failed with RuntimeError; skipping'.format(st))
            bar.update(1)
            continue

# %% [markdown]
# Save all the fits; use `load_state_fits` (untested) to load them back in.

# %%
def save_state_fits(fits, directory):
    try:
        os.makedirs(directory)
    except OSError:
        pass
    
    for k, f in fits.items():
        fname = os.path.join(directory, k + '.nc')
        fname_tmp = fname + '.temp'
        az.to_netcdf(f, fname_tmp)
        os.rename(fname_tmp, fname)
def load_state_fits(directory):
    fits = {}
    for f in glob.glob(os.path.join(directory, '*.nc')):
        k = os.path.splitext(os.path.split(f)[1])[0]
        fits[k] = az.from_netcdf(f)


# %%
save_state_fits(state_fits, 'state_fits_US')

# %% [markdown]
# And make a plot that is similar to the one at http://rt.live

# %%
#import glob
#state_fits = load_state_fits('state_fits_EUW')


# %%
# set customized palette
sns.set_palette(rtlive)

# %%
nc = 4
nr = 11
# temporarily limit states for plot testing
# state_count = 12

fig, axes = subplots(nrows=nr, ncols=nc, figsize=(24, 5*nr))

count = 0;
for idx, ((k,fit), ax) in enumerate(zip(state_fits.items(), axes.flatten())):
    
    # if count >= state_count:
    #    break

    m = median(fit.posterior.Rt, axis=(0,1))
    hh = percentile(fit.posterior.Rt, 97.5, axis=(0,1))
    h = percentile(fit.posterior.Rt, 84, axis=(0,1))
    l = percentile(fit.posterior.Rt, 16, axis=(0,1))
    ll = percentile(fit.posterior.Rt, 2.5, axis=(0,1))

    x = fit.posterior.Rt_dates.values
        line_1 = np.ones(len(x))

    # NOTE: try this approach instead (filling with NaN):
    m_good = m.copy()
    m_bad = m.copy()
    m_good[m_good > 1.30] = np.nan
    m_bad[m_bad <= 0.98] = np.nan
    
    ax.plot(x, m_good, ls='-', marker='', lw=1.5, color=sns.color_palette()[0])
    ax.plot(x, m_bad, ls='-', marker='', lw=1.5, color=sns.color_palette()[1])

    good_dates = m <= 1.0
    bad_dates = m > 1.0

    #ax.plot_date(x[good_dates], m[good_dates], ls='-', marker='', xdate=True, color=sns.color_palette()[0])
    #ax.fill_between(x, where(h<1, h, 1) , where(l<1, l, 1), alpha=0.10, color=sns.color_palette()[0])
    
    ax.fill_between(x, where(hh<1, hh, 1) , where(ll<1, ll, 1), alpha=0.10, color=sns.color_palette()[0])
    
    #ax.plot_date(x[bad_dates], m[bad_dates], ls='-', marker='', xdate=True, color=sns.color_palette()[1])
    #ax.fill_between(x, where(h>1, h, 1) , where(l>1, l, 1), alpha=0.10, color=sns.color_palette()[1])
    
    ax.fill_between(x, where(hh>1, hh, 1) , where(ll>1, ll, 1), alpha=0.10, color=sns.color_palette()[1])
    
    #ax.plot_date(x, m, ls='-', marker='', xdate=True, color=sns.color_palette()[0])
    ax.plot(x, line_1, ls='dotted', color = sns.xkcd_rgb["light grey"], lw=3)

    date_fmt = md.DateFormatter('%m-%d')
    ax.xaxis.set_major_formatter(date_fmt)
    
    ax.set_title(k)
    ax.set_ylabel(r'$R_t$')

    ax.set_xlim([date_then, date_now])

    ax.set_ylim(bottom = 0,
                top = 3,
                auto = False)
    sns.despine()
    #y0, y1 = ax.get_ylim()
    #if y0 < 0:
    #    ax.set_ylim(0, y1)
    
    count +=1

else:
    [ax.set_visible(False) for ax in axes.flatten()[idx+1:]]

# %%
# reset to orig palette
sns.set_palette('colorblind')

# %% [markdown]
# Here is a nice plot that shows the distribution of $R_t$ at the latest day of sampling, ordered by its median value:

# %%
nc = 4
nd = 1000

d = {'state': [], 'Rt': []}
for k, f in state_fits.items():
    d = {'state': concatenate((d['state'], (k,)*(nc*nd))), 
         'Rt': concatenate((d['Rt'], f.posterior.Rt[:,:,-1].values.flatten()))}
df = pd.DataFrame(d)

sort_Rts = [median(fit.posterior.Rt[:,:,-1]) for fit in state_fits.values()]
state_order = array(list(state_fits.keys()))[argsort(sort_Rts)]

# %% [markdown]
# Compare to http://rt.live:

# %%
figure(figsize=(24,4))
sns.boxplot(x='state', y='Rt', data=df, order=state_order, fliersize=0, whis=1.0/1.35, )
axhline(1, color='k')
xticks(rotation=90)
axis(ymin=0, ymax=2.5)

# %% [markdown]
# Should check in on this model for $\tau$---in the states where it is well-measured, we favor a slightly smaller $\tau$ than the default prior.  Might be worth building a hierarchical model to re-infer $\tau$ from the national data....  Black line is the prior.

# %%
with sns.color_palette('husl', n_colors=len(state_fits)):
    for f in state_fits.values():
        sns.kdeplot(f.posterior.tau.values.flatten())
xs = linspace(0, 20, 1024)
plot(xs, ss.lognorm(0.57, scale=exp(1.4)).pdf(xs), color='k')
axis(xmin=0,xmax=12)

xlabel(r'$\tau$ ($\mathrm{d}$)')

