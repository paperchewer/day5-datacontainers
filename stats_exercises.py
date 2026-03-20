#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:07:46 2026

@author: tinca775
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# a. discrete RV poisson distribution 
mu = 3  # mean of poisson

poisson = stats.poisson(mu)
x_discrete = np.arange(0, 15)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Poisson Distribution (mu=3)")

# PMF
axes[0].bar(x_discrete, poisson.pmf(x_discrete))
axes[0].set_title("PMF")
axes[0].set_xlabel("k")

# CDF
axes[1].bar(x_discrete, poisson.cdf(x_discrete))
axes[1].set_title("CDF")
axes[1].set_xlabel("k")

# Histogram of 1000 random realizations
axes[2].hist(poisson.rvs(1000), bins=15)
axes[2].set_title("Histogram (1000 samples)")
axes[2].set_xlabel("k")

plt.tight_layout()
plt.savefig("poisson.png")
plt.show()


# b. Normal distribution continuous 

normal = stats.norm(loc = 0, scale =1) # mean 0, variance 1
x_cont = np.linspace(-4,4,100)

fig, axes = plt.subplots(1,3, figsize=(15,4))
fig.suptitle("normal distribution mean = 0 variance =1")

# PDF (for continuous we use PDF not PMF)
axes[0].plot(x_cont, normal.pdf(x_cont))
axes[0].set_title("PDF")
axes[0].set_xlabel("x")

# CDF
axes[1].plot(x_cont, normal.cdf(x_cont))
axes[1].set_title("CDF")
axes[1].set_xlabel("x")

# Histogram of 1000 random realizations
axes[2].hist(normal.rvs(1000), bins=30)
axes[2].set_title("Histogram (1000 samples)")
axes[2].set_xlabel("x")

plt.tight_layout()
plt.savefig("normal.png")
plt.show()



#c. Test if two sets of (independent) random data comes from the same distribution

# Two samples from the SAME distribution
data1 = stats.norm.rvs(loc=0, scale=1, size=1000)
data2 = stats.norm.rvs(loc=0, scale=1, size=1000)

# Two samples from DIFFERENT distributions
data3 = stats.norm.rvs(loc=5, scale=1, size=1000)

t_stat, p_value = stats.ttest_ind(data1, data2)
print("Same distribution:")
print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
print(f"  Same distribution? {p_value > 0.05}")

t_stat2,p_value2 = stats.ttest_ind(data1, data3)
print("\nDifferent distribution")
print(f"  t-statistic: {t_stat2:.4f}, p-value: {p_value2:.4f}")
print(f" Same distribution? {p_value2 > 0.05}")
