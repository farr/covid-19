# Measuring COVID-19 real-time transmission rate (Rt)
A collection of work related to COVID-19

These are modified versions of [Kevin Systrom's](https://github.com/k-sys/covid-19/) and [William Farr's](https://github.com/farr/covid-19/) models for estimating Covid-19 Rt (transmission rate of Covid-19 in real-time.) Nothing is changed in the way the models themselves operate.
### Modifications
- method for dealing with negative case count on consequtive days (which, of course, is an impossibility, caused by non-linear case reporting and/or errors in reports
- layout implemented and streamlined across notebooks, now using the color scheme featured on https://rt.live
- plotting function for Stan R0 notebooks
- Stan notebooks rewritten to use pyplot + numPy instead of pyLab
- extended save and load functions for processed data
- models processing EU data as well as US data, with possibility of expanding to any country reported on https://ourworldindata.org
- high resolution png files of the last runs (residing in `image`folder)

### Notebooks
These are the notebooks containing the aforementioned modifications:
- [Realtime Rt mcmc EUW.ipynb](https://github.com/pati610/covid-19/blob/master/Realtime%20Rt%20mcmc%20EUW.ipynb)
- [Realtime Rt mcmc US.ipynb](https://github.com/pati610/covid-19/blob/master/Realtime%20Rt%20mcmc%20US.ipynb)
- [Stan R0 EUW.ipynb](https://github.com/pati610/covid-19/blob/master/Stan%20R0%20EUW.ipynb)
- [Stan R0 US.ipynb](https://github.com/pati610/covid-19/blob/master/Stan%20R0%20US.ipynb)

The original notebooks have not been modified.
