#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from six import print_
import numpy as np

def squash_timed_labels(start_times, end_times, labels):
    centre_times = np.mean((start_times, end_times), axis=0)
    centre_crossovers = centre_times[:-1] + np.diff(centre_times) / 2
    disjoint_starts = np.hstack((centre_times[0], centre_crossovers))
    disjoint_ends = np.hstack((centre_crossovers, centre_times[-1]))
    change_points = labels[1:] != labels[:-1]
    return disjoint_starts[np.hstack(([True],change_points))], disjoint_ends[np.hstack((change_points, [True]))], labels[np.hstack(([True],change_points))]

from os.path import dirname
from madmom.features.chords import write_chords
def write_chord_file(chord_file_path, start_times, end_times, chord_labels):
    squashed_start_times, squashed_end_times, squashed_chord_labels = squash_timed_labels(start_times, end_times, chord_labels)
    chord_file_content = np.array([(x,y,z) for x,y,z in zip(squashed_start_times, squashed_end_times, squashed_chord_labels)], dtype=[('start', squashed_start_times.dtype), ('end', squashed_end_times.dtype), ('label', squashed_chord_labels.dtype)])
    try:
        os.makedirs(dirname(chord_file_path))
    except OSError:
        pass
    write_chords(chord_file_content, chord_file_path)

import pandas as pd
def run_on_file_list(relative_list_path, callable_object, verbose=False):
    relative_list = pd.read_csv(relative_list_path, header=None, names=['Path'])
    output = []
    for relative_path in relative_list['Path']:
        if verbose:
            msg = 'Processing ' + relative_path
            print_(msg, end='', flush=True)
        output.append(callable_object(relative_path))
        if verbose:
            print_('\r' + ' ' * len(msg) + '\r', end='', flush=True)
    if output != [None] * len(output):
        return output
        
def run_on_file_list_with_arg(relative_list_path, callable_object, arguments, verbose=False):
    relative_list = pd.read_csv(relative_list_path, header=None, names=['Path'])
    output = []
    for relative_path, arg in zip(relative_list['Path'], arguments):
        if verbose:
            msg = 'Processing ' + relative_path
            print_(msg, end='', flush=True)
        output.append(callable_object(relative_path, arg))
        if verbose:
            print_('\r' + ' ' * len(msg) + '\r', end='', flush=True)
    if output != [None] * len(output):
        return output

from scipy.linalg import circulant
import itertools
from os.path import join, isfile
from hiddini import ObservationsTemplateCosSim

class ChordsFromTemplates(object):
    def __init__(self, audio_dir, chordfile_dir, chromas, chordtypes, type_templates, chroma_extractor, audio_suffix='.wav', chordfile_suffix='.lab'):
        self.audio_dir = audio_dir
        self.chordfile_dir = chordfile_dir
        self.chords = np.array([':'.join(x) for x in itertools.product(chromas, chordtypes)])
        self.chroma_extractor = chroma_extractor
        self.audio_suffix = audio_suffix
        self.chordfile_suffix = chordfile_suffix
        chord_templates = np.dstack([circulant(i) for i in type_templates]).reshape(len(chromas), -1).T
        self.observer = ObservationsTemplateCosSim(chord_templates)
    
    def __call__(self, relative_path, timed_chromagram=None):
        if timed_chromagram is None:
            chromagram, (start_times, end_times) = self.chroma_extractor(join(self.audio_dir, relative_path+self.audio_suffix))
        else:
            chromagram, (start_times, end_times) = timed_chromagram
        return start_times, end_times, self.observer(chromagram.T)

class FramewiseChordsFromTemplates(ChordsFromTemplates):
    def __init__(self, audio_dir, chordfile_dir, chromas, chord_types, type_templates, chroma_extractor, audio_suffix='.wav', chordfile_suffix='.lab', overwrite=False):
        super(FramewiseChordsFromTemplates, self).__init__(audio_dir, chordfile_dir, chromas, chord_types, type_templates, chroma_extractor, audio_suffix, chordfile_suffix)
        self.overwrite = overwrite
    
    def __call__(self, relative_path, timed_chromagram=None):
        chordfile_path = join(self.chordfile_dir, relative_path+self.chordfile_suffix)
        if not isfile(chordfile_path) or self.overwrite:
            start_times, end_times, chord_probs = super(FramewiseChordsFromTemplates, self).__call__(relative_path, timed_chromagram)
            max_indices = np.argmax(chord_probs, axis=0)
            framewise_chord_labels = self.chords[max_indices]
            write_chord_file(chordfile_path, start_times, end_times, framewise_chord_labels)
            num_frames = chord_probs.shape[1]
            return np.sum(np.log(chord_probs[max_indices, range(num_frames)])), num_frames

from hiddini import HMMRaw
class HMMSmoothedChordsFromTemplates(ChordsFromTemplates):
    def __init__(self, audio_dir, chordfile_dir, chromas, chord_types, type_templates, chroma_extractor, chord_self_prob, audio_suffix='.wav', chordfile_suffix='.lab', overwrite=False, decode_method='decodeMAP'):
        super(HMMSmoothedChordsFromTemplates, self).__init__(audio_dir, chordfile_dir, chromas, chord_types, type_templates, chroma_extractor, audio_suffix, chordfile_suffix)
        self.overwrite = overwrite
        num_chords = len(self.chords)
        trans_prob = np.full((num_chords, num_chords), (1-chord_self_prob)/(num_chords-1))
        np.fill_diagonal(trans_prob, chord_self_prob)
        hmm = HMMRaw(trans_prob, np.full(num_chords, 1/num_chords))
        self.decode = getattr(hmm, decode_method)
    
    def __call__(self, relative_path, timed_chromagram=None):
        chordfile_path = join(self.chordfile_dir, relative_path+self.chordfile_suffix)
        if not isfile(chordfile_path) or self.overwrite:
            start_times, end_times, chord_probs = super(HMMSmoothedChordsFromTemplates, self).__call__(relative_path, timed_chromagram)
            hmm_smoothed_state_indices, log_prob, confidence = self.decode(chord_probs)
            write_chord_file(chordfile_path, start_times, end_times, self.chords[hmm_smoothed_state_indices])
            return log_prob, confidence

from subprocess import check_output, CalledProcessError
import re
import os.path
def evaluate_chords(list_path, reference_dir, test_dir, preset, output_path, reference_suffix='.lab', test_suffix='.lab', overwrite=False):
    if overwrite or not os.path.isfile(output_path):
        try:
            check_output(['MusOOEvaluator', '--chords', preset, '--list', list_path, '--refdir', reference_dir, '--testdir', test_dir, '--refext', reference_suffix, '--testext', test_suffix, '--output', output_path, '--csv'])
        except OSError as e:
            if e.errno == 2:
                print('\n\033[91m' + 'MusOOEvaluator cannot be found in the current path' + '\033[0m\n')
            raise e
        except CalledProcessError as e:
            print('\033[91m' + e.output + '\033[0m')
            raise e
    with open(output_path) as output_file:
        match = re.search('Average score: ([\d\.]+)%', output_file.read())
        return float(match.group(1))
        
