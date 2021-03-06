#!/usr/bin/env python3
"""Prototype of the differential operator sampling method."""
import time
import multiprocessing as mp
import numpy as np
import pandas as pd


# Minimal data set required
groups = range(1, 1+1)
adjacent_cell = {'inner': 'outer', 'outer': 'inner'}
xs = {
    'inner': {
        1: {
            'scatter': {1: 1.0},
            'capture': 1.0,},},}
rxn_probs = {}
xs_totals = {}
for c in xs:
    rxn_probs[c] = {}
    xs_totals[c] = {}
    for g in xs[c]:
        scatter_xs = sum(xs[c][g]['scatter'].values())
        capture_xs = xs[c][g]['capture']
        rxn_probs[c][g] = {'scatter': scatter_xs, 'capture': capture_xs}
        xs_totals[c][g] = scatter_xs + capture_xs

def get_isotropic():
    """
    Sample a point on the unit sphere istotropically.
    """
    costheta = 2. * np.random.random() - 1. # mu
    sintheta = np.sqrt(1. - costheta * costheta)
    phi = 2. * np.pi * np.random.random()
    cosphi = np.cos(phi)
    sinphi = np.sqrt(1. - cosphi * cosphi)
    return np.array([costheta, sintheta * cosphi, sintheta * sinphi])

def distance_to_sphere(pos, dof, radius):
    """Distance between particle and origin-centered sphere."""
    # Compute discriminant
    dir_dot_pos = np.dot(dof, pos)
    discriminant = np.square(dir_dot_pos) - np.dot(pos, pos) + np.square(radius)
    if discriminant < 0.:
        # line and sphere do not intersect
        return np.inf
    if discriminant > 0.:
        # two solutions exist, return smallest positive
        value = -dir_dot_pos + np.sqrt(discriminant)
        return value if value > 0 else np.inf
    # line is tangent to sphere
    raise RuntimeError

def simulate_particle(history):
    """Simulate the particle history."""
    # Set up estimators (capture)
    estimators = {
        g: {'capture': 0, 'capture sqr': 0, 'd_capture': 0, 'd_capture sqr': 0}
        for g in groups}
    # Run simulation
    np.random.seed(history)
    pos = np.zeros(3)
    dof = get_isotropic()
    group = 1
    cell = 'inner'
    k = 1. / (4. * np.pi) # The probability of reaching current state
    d_k = 0. # Derivative of k w.r.t x
    while True:
        # Sample next event
        total_xs = xs_totals[cell][group]
        distances = {
            'collide': (- np.log(np.random.random()) / total_xs),
            'cross_surf_1': distance_to_sphere(pos, dof, 10.0)}
        min_event = min(distances, key=distances.get)
        distance = distances[min_event]
        pos += dof * (distance + 1e-16)
        # Precompute probability of streaming to point
        transmit_prob = total_xs / np.square(distance) * np.exp(-total_xs * distance)
        # Update particle to new state
        if min_event == 'collide':
            # Sample reaction
            threshold = np.random.random() * total_xs
            current = 0.
            for reaction in rxn_probs[cell][group]:
                reaction_xs = rxn_probs[cell][group][reaction]
                current += reaction_xs
                if threshold < current:
                    if reaction == 'scatter':
                        # Determine outgoing group
                        threshold = np.random.random() * reaction_xs
                        current = 0.
                        for g_out, val in xs[cell][group]['scatter'].items():
                            current += val
                            if threshold < current:
                                group = g_out
                                break
                        # Scatter istotropically
                        dof = get_isotropic()
                        # Update probability of moving to state and derivative thereof
                        transition_prob = transmit_prob * reaction_xs / total_xs / (4. * np.pi)
                        d_transition_prob = (
                            - transmit_prob * distance * reaction_xs / total_xs
                            / (4. * np.pi))
                        d_k = transition_prob * d_k + k * d_transition_prob
                        k = k * transition_prob
                        break
                    if reaction == 'capture':
                        # Score estimators
                        estimators[group]['capture'] += 1
                        estimators[group]['capture sqr'] += 1
                        # Update probability of moving to state and derivative thereof
                        transition_prob = transmit_prob * reaction_xs / total_xs
                        d_transition_prob = transmit_prob * (1. - reaction_xs * distance) / total_xs
                        d_k = transition_prob * d_k + k * d_transition_prob
                        k = k * transition_prob
                        dos_score = d_k / k
                        estimators[group]['d_capture'] += dos_score
                        estimators[group]['d_capture sqr'] += dos_score * dos_score
                        return estimators
                    raise RuntimeError
        elif min_event == 'cross_surf_1':
            # # Score estimators
            # estimators[group]['leak'] += 1
            # estimators[group]['leak sqr'] += 1
            # # Update probability of moving to state and derivative thereof
            # transition_prob = transmit_prob / total_xs
            # d_transition_prob = - transmit_prob * distance / total_xs
            # d_k = transition_prob * d_k + k * d_transition_prob
            # k = k * transition_prob
            # dos_score = d_k / k
            # estimators[group]['d_leak'] += dos_score
            # estimators[group]['d_leak sqr'] += dos_score * dos_score
            # Update particle state
            cell = adjacent_cell[cell]
            cell = 'void'
            return estimators
        else:
            raise RuntimeError


def run_histories(runs, procs):
    """Runs runs runs on procs processes."""
    tick = time.perf_counter()
    print(f"Running {runs} histories on {procs} processes...")
    sums = {
        group: {'capture': 0, 'capture sqr': 0, 'd_capture': 0, 'd_capture sqr': 0}
        for group in groups}
    output = {}
    with mp.Pool(processes=procs) as pool:
        results = pool.imap(simulate_particle, range(runs), chunksize=1000)
        for result in results:
            for group in groups:
                sums[group]['capture'] += result[group]['capture']
                sums[group]['capture sqr'] += result[group]['capture sqr']
                sums[group]['d_capture'] += result[group]['d_capture']
                sums[group]['d_capture sqr'] += result[group]['d_capture sqr']
        for group in sums:
            capture_mean = sums[group]['capture'] / runs
            capture_mean_square = sums[group]['capture sqr'] / runs
            d_capture_mean = sums[group]['d_capture'] / runs
            d_capture_mean_square = sums[group]['d_capture sqr'] / runs
            output[group] = {
                'capture': capture_mean,
                'capture stdev': np.sqrt((capture_mean_square - capture_mean ** 2) / runs),
                'd_capture': d_capture_mean,
                'd_capture stdev': np.sqrt((d_capture_mean_square - d_capture_mean ** 2) / runs),
                }
    tock = time.perf_counter()
    print(f"time elapsed: {tock-tick:0.4f}")
    return output



if __name__ == '__main__':
    # Run RUNS
    print(run_histories(10000000, 8))
