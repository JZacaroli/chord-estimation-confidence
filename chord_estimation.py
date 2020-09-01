#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from six import print_
import numpy as np
import madmom
from madmom.audio.chroma import *
from scipy.signal import medfilt
from madmom.io.audio import load_wave_file

class MadMomChromaExtractor(object):
    def __init__(self, samplerate, frame_size, step_size):
        self.samplerate = samplerate
        self.frame_size = frame_size
        self.step_size = step_size

    def get_frame_times(self, chromagram):
        start_times = self.step_size / self.samplerate * np.arange(len(chromagram)) - self.frame_size/(2*self.samplerate)
        return start_times, start_times+self.frame_size/self.samplerate

class MadMomDeepChromaExtractor(MadMomChromaExtractor):
    def __init__(self, samplerate, frame_size, step_size):
        super(MadMomDeepChromaExtractor, self).__init__(samplerate, frame_size, step_size)
        if samplerate != 44100 or frame_size != 8192 or step_size != 4410:
            raise ValueError('Parameter values not supported')
        self.extractor = madmom.audio.chroma.DeepChromaProcessor(num_channels=1)

    def __call__(self, audiopath):
        chromagram = self.extractor(audiopath)
        chromagram = np.roll(chromagram, 3, axis=1)
        return chromagram, self.get_frame_times(chromagram)

class MadMomCLPChromaExtractor(MadMomChromaExtractor):
    def __init__(self, samplerate, frame_size, step_size, apply_smoothing=False):
        super(MadMomCLPChromaExtractor, self).__init__(samplerate, frame_size, step_size)
        self.apply_smoothing = apply_smoothing

    def __call__(self, audiopath):
        chromagram = madmom.audio.chroma.CLPChroma(audiopath, fps=self.samplerate/self.step_size)
        chromagram = np.roll(chromagram, 3, axis=1)
        if self.apply_smoothing:
            chromagram = medfilt(chromagram, kernel_size=[21,1])
        return chromagram, self.get_frame_times(chromagram)

class MadMomPCPChromaExtractor(MadMomChromaExtractor):
    def __init__(self, samplerate, frame_size, step_size):
        super(MadMomPCPChromaExtractor, self).__init__(samplerate, frame_size, step_size)

    def __call__(self, audiopath):
        chromagram = madmom.audio.chroma.PitchClassProfile(audiopath, num_channels=1, sample_rate=self.samplerate, frame_size=self.frame_size, hop_size=self.step_size, num_classes=12)
        return chromagram, self.get_frame_times(chromagram)

class MadMomHPCPChromaExtractor(MadMomChromaExtractor):
    def __init__(self, samplerate, frame_size, step_size):
        super(MadMomHPCPChromaExtractor, self).__init__(samplerate, frame_size, step_size)

    def __call__(self, audiopath):
        chromagram = madmom.audio.chroma.HarmonicPitchClassProfile(audiopath, num_channels=1, sample_rate=self.samplerate, frame_size=self.frame_size, hop_size=self.step_size, num_classes=12)
        return chromagram, self.get_frame_times(chromagram)


from librosa.feature.spectral import chroma_cens
from librosa import load as librosa_load

class LibrosaCENSChromaExtractor(MadMomChromaExtractor):
    """ Technically not a *madmom* chroma extractor """
    def __init__(self, samplerate, frame_size, step_size):
        super(LibrosaCENSChromaExtractor, self).__init__(samplerate, frame_size, step_size)

    def __call__(self, audiopath):
        audio,_ = librosa_load(audiopath, sr=self.samplerate, mono=True)
        chromagram = chroma_cens(y=audio, sr=self.samplerate, hop_length=self.step_size)
        chromagram = np.swapaxes(chromagram, 0, 1)
        chromagram = np.roll(chromagram, 3, axis=1)
        return chromagram, self.get_frame_times(chromagram)

class RmsEnergyExtractor():
    def __init__(self, frame_size, step_size, sample_rate):
        self.frame_size = frame_size
        self.step_size = step_size
        self.sample_rate = sample_rate

    def __call__(self, audiopath):
        audio = load_wave_file(audiopath, sample_rate=self.sample_rate)[0]
        rms_values = np.zeros(1+int(np.round(len(audio)/self.step_size))) - 1
        frame_start=0
        block_num = 0
        while (frame_start + self.step_size) <= len(audio):
            audio_block = np.array(audio[frame_start:frame_start+self.frame_size], dtype='int64')
            rmsVal = np.sqrt(np.mean(np.square(audio_block)))
            rms_values[block_num] = rmsVal
            frame_start += self.step_size
            block_num += 1
        return rms_values


def squash_timed_labels(start_times, end_times, labels):
    centre_times = np.mean((start_times, end_times), axis=0)
    centre_crossovers = centre_times[:-1] + np.diff(centre_times) / 2
    disjoint_starts = np.hstack((centre_times[0], centre_crossovers))
    disjoint_ends = np.hstack((centre_crossovers, centre_times[-1]))
    change_points = labels[1:] != labels[:-1]
    return disjoint_starts[np.hstack(([True],change_points))], disjoint_ends[np.hstack((change_points, [True]))], labels[np.hstack(([True],change_points))]

from os.path import dirname
from madmom.io import write_chords
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

    return output

def run_on_file_list_with_arg(relative_list_path, callable_object, arguments, verbose=False):
    relative_list = pd.read_csv(relative_list_path, header=None, names=['Path'])
    output = []
    for relative_path, arg in zip(relative_list['Path'], arguments):
        if verbose:
            msg = 'Processing ' + relative_path
            print_(msg, end='', flush=True)

        x = callable_object(relative_path, arg)
        output.append(x)
        if verbose:
            print_('\r' + ' ' * len(msg) + '\r', end='', flush=True)
    return output

from scipy.linalg import circulant
import itertools
from os.path import join, isfile
from hiddini import ObservationsTemplateCosSim

class ChordsFromTemplates(object):
    def __init__(self, audio_dir, chordfile_dir, chromas, chordtypes, type_templates, chroma_extractor, audio_suffix='.wav', chordfile_suffix='.lab', overwrite=False, root_type_separator=':'):
        self.audio_dir = audio_dir
        self.chordfile_dir = chordfile_dir
        self.chords = np.array([root_type_separator.join(x) for x in itertools.product(chromas, chordtypes)])
        self.chroma_extractor = chroma_extractor
        self.audio_suffix = audio_suffix
        self.chordfile_suffix = chordfile_suffix
        self.overwrite = overwrite
        chord_templates = np.dstack([circulant(i) for i in type_templates]).reshape(len(chromas), -1).T
        self.observer = ObservationsTemplateCosSim(chord_templates)

    def __call__(self, relative_path, timed_chromagram=None):
    # Work out the observation probabilities
        if timed_chromagram is None:
            # Create time series of chromas
            chromagram, (start_times, end_times) = self.chroma_extractor(join(self.audio_dir, relative_path+self.audio_suffix))
        else:
            chromagram, (start_times, end_times) = timed_chromagram
        # Observer() returns probabilities of chord states at each time step
        return start_times, end_times, self.observer(chromagram.T)

class FramewiseChordsFromTemplates(ChordsFromTemplates):
    def __init__(self, audio_dir, chordfile_dir, chromas, chord_types, type_templates, chroma_extractor, audio_suffix='.wav', chordfile_suffix='.lab', overwrite=False):
        super(FramewiseChordsFromTemplates, self).__init__(audio_dir, chordfile_dir, chromas, chord_types, type_templates, chroma_extractor, audio_suffix, chordfile_suffix, overwrite)

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
    def __init__(self, audio_dir, chordfile_dir, chromas, chord_types, type_templates, chroma_extractor, chord_self_prob, audio_suffix='.wav', chordfile_suffix='.lab', overwrite=False, decode_method='decodeMAP', is_entropy=False):
        super(HMMSmoothedChordsFromTemplates, self).__init__(audio_dir, chordfile_dir, chromas, chord_types, type_templates, chroma_extractor, audio_suffix, chordfile_suffix, overwrite)
        num_chords = len(self.chords)
        trans_prob = np.full((num_chords, num_chords), (1-chord_self_prob)/(num_chords-1))
        np.fill_diagonal(trans_prob, chord_self_prob)
        hmm = HMMRaw(trans_prob, np.full(num_chords, 1/num_chords))
        self.decode = getattr(hmm, decode_method)
        self.is_entropy=is_entropy

    def __call__(self, relative_path, timed_chromagram=None):
        chordfile_path = join(self.chordfile_dir, relative_path+self.chordfile_suffix)
        #Only do HMM smoothing if we've not already done it
        if not isfile(chordfile_path) or self.overwrite:
            # Calculate the framewise likelihood of each chord
            start_times, end_times, chord_probs = super(HMMSmoothedChordsFromTemplates, self).__call__(relative_path, timed_chromagram)
            # Viterbi algorithm to decode the smoothed states, their probabilities and confidences.
            retvals = self.decode(chord_probs) #First is the optimal state sequence
            write_chord_file(chordfile_path, start_times, end_times, self.chords[retvals[0]])
            if self.is_entropy:
                return retvals[1]
            return retvals[1], retvals[2]

from subprocess import check_output, CalledProcessError
import re
import os.path
def evaluate_chords(list_path, reference_dir, test_dir, preset, output_path, reference_suffix='.lab', test_suffix='.lab', overwrite=False):
    print('Output path:', output_path)
    if overwrite or not os.path.isfile(output_path):
        try:
            check_output(['./MusOOEvaluator', '--chords', preset, '--list', list_path, '--refdir', reference_dir, '--testdir', test_dir, '--refext', reference_suffix, '--testext', test_suffix, '--output', output_path, '--csv'])
        except OSError as e:
            if e.errno == 2:
                print('\n\033[91m' + 'MusOOEvaluator cannot be found in the current path' + '\033[0m\n')
            raise e
        except CalledProcessError as e:
            print(e.output)
            raise e
    with open(output_path) as output_file:
        match = re.search('Average score: ([\d\.]+)%', output_file.read())
        return float(match.group(1))
