#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
import madmom
from os.path import join, basename, splitext, isfile
from chord_estimation import HMMSmoothedChordsFromTemplates, run_on_file_list, run_on_file_list_with_arg, evaluate_chords
import pandas as pd

def run_experiments(list_path, audio_dir, reference_label_dir, output_dir, chroma_extractor, chromas, chord_types, type_templates, self_probs, audio_suffix='.wav', chordfile_suffix='.lab', hmm_decoders=['decodeMAP']):
    list_name = splitext(basename(list_path))[0]
    
    if not all([isfile(join(output_dir, 'HMMChords-{}-{}Ps'.format(y, round(x,3)), '{}-ResultsMirex.txt'.format(list_name))) for x in self_probs for y in hmm_decoders]):
        # Calculate chromagrams
        chromagrams = run_on_file_list(list_path, lambda x: chroma_extractor(join(audio_dir, x+audio_suffix)), verbose=True)
        # Calculate chords with confidence from chromagrams
        print('HMM self probabilities: ', end='')
        for self_prob in self_probs:
            print(round(self_prob, 3), end=' ')
            for decoder in hmm_decoders:
                chord_dir = join(output_dir, 'HMMChords-{}-{}Ps'.format(decoder, round(self_prob,3)))
                log_probs_and_confidences = run_on_file_list_with_arg(list_path, HMMSmoothedChordsFromTemplates(audio_dir, chord_dir, chromas, chord_types, type_templates, chroma_extractor, self_prob, audio_suffix, chordfile_suffix, decode_method=decoder), chromagrams)
                if log_probs_and_confidences:
                    pd.DataFrame(log_probs_and_confidences).to_csv(join(chord_dir, list_name+'-logprobs_confidences.csv'), index=False, header=False)
        print('')
    else:
        print('HMM self probabilities: ' + ' '.join(map(lambda x: str(round(x, 3)), self_probs)))
    # Evaluate chord sequences
    for decoder in hmm_decoders:
        hmm_scores = []
        for self_prob in self_probs:
            chord_dir = join(output_dir, 'HMMChords-{}-{}Ps'.format(decoder, round(self_prob, 3)))
            hmm_scores.append(evaluate_chords(list_path, reference_label_dir, chord_dir, 'MirexMajMin', join(chord_dir, '{}-ResultsMirex.txt'.format(list_name))))
        print('{} {} HMM chord scores: {}'.format(list_name, decoder, hmm_scores))
        if len(self_probs) > 1:
            print('Best score of {} for self probability of {}'.format(np.max(hmm_scores), self_probs[np.argmax(hmm_scores)]))


class MadMomChromaExtractorShim(object):
    def __init__(self, samplerate, frame_size, step_size):
        self.samplerate = samplerate
        self.frame_size = frame_size
        self.step_size = step_size
    
    def get_frame_times(self, chromagram):
        start_times = self.step_size / self.samplerate * np.arange(len(chromagram)) - self.frame_size/(2*self.samplerate)
        return start_times, start_times+self.frame_size/self.samplerate

class MadMomDeepChromaExtractorShim(MadMomChromaExtractorShim):
    def __init__(self, samplerate, frame_size, step_size):
        super(MadMomDeepChromaExtractorShim, self).__init__(samplerate, frame_size, step_size)
        if samplerate != 44100 or frame_size != 8192 or step_size != 4410:
            raise ValueError('Parameter values not supported')
        self.extractor = madmom.audio.chroma.DeepChromaProcessor(num_channels=1)
    
    def __call__(self, audiopath):
        chromagram = self.extractor(audiopath)
        chromagram = np.roll(chromagram, 3, axis=1)
        return chromagram, self.get_frame_times(chromagram)

class MadMomCLPChromaExtractorShim(MadMomChromaExtractorShim):
    def __init__(self, samplerate, frame_size, step_size):
        super(MadMomCLPChromaExtractorShim, self).__init__(samplerate, frame_size, step_size)
    
    def __call__(self, audiopath):
        chromagram = madmom.audio.chroma.CLPChroma(audiopath, fps=self.samplerate/self.step_size)
        chromagram = np.roll(chromagram, 3, axis=1)
        return chromagram, self.get_frame_times(chromagram)
        
class MadMomPCPChromaExtractorShim(MadMomChromaExtractorShim):
    def __init__(self, samplerate, frame_size, step_size):
        super(MadMomPCPChromaExtractorShim, self).__init__(samplerate, frame_size, step_size)
    
    def __call__(self, audiopath):
        chromagram = madmom.audio.chroma.PitchClassProfile(audiopath, num_channels=1, sample_rate=self.samplerate, frame_size=self.frame_size, hop_size=self.step_size, num_classes=12)
        return chromagram, self.get_frame_times(chromagram)
        
class MadMomHPCPChromaExtractorShim(MadMomChromaExtractorShim):
    def __init__(self, samplerate, frame_size, step_size):
        super(MadMomHPCPChromaExtractorShim, self).__init__(samplerate, frame_size, step_size)
    
    def __call__(self, audiopath):
        chromagram = madmom.audio.chroma.HarmonicPitchClassProfile(audiopath, num_channels=1, sample_rate=self.samplerate, frame_size=self.frame_size, hop_size=self.step_size, num_classes=12)
        return chromagram, self.get_frame_times(chromagram)
        

## Chroma extraction configuration
samplerate = 44100
block_size = 8192
step_size = 4410

## Chord estimation configuration
type_templates = np.array([[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]])
chord_types = ['maj', 'min', 'dim', 'aug']
chromas = ['A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab']
chord_self_probs = [0.1]
hmm_decoders = ['decodeMAP_with_medianOPC', 'decode_with_PPD']

## Experiments
run_experiments('Lists/sines.lst', 'Audio', 'Ground-Truth', join('Experiments', 'DeepChroma', 'Sines'), MadMomDeepChromaExtractorShim(samplerate, block_size, step_size), chromas, chord_types, type_templates, self_probs=chord_self_probs, hmm_decoders=hmm_decoders)

# Isophonics
# run_experiments('Lists/IsophonicsChords2010.lst', '/import/c4dm-datasets/C4DM Music Collection', '../Ground-Truth/Chord Annotations C4DM', join('Experiments', 'DeepChroma', 'Isophonics'), MadMomDeepChromaExtractorShim(samplerate, block_size, step_size), chromas, chord_types, type_templates, self_probs=chord_self_probs, hmm_decoders=hmm_decoders)
# run_experiments('Lists/IsophonicsChords2010.lst', '/import/c4dm-datasets/C4DM Music Collection', '../Ground-Truth/Chord Annotations C4DM', join('Experiments', 'CLPChroma', 'Isophonics'), MadMomCLPChromaExtractorShim(samplerate, block_size, step_size), chromas, chord_types, type_templates, self_probs=chord_self_probs, hmm_decoders=hmm_decoders)
# run_experiments('Lists/IsophonicsChords2010.lst', '/import/c4dm-datasets/C4DM Music Collection', '../Ground-Truth/Chord Annotations C4DM', join('Experiments', 'PCPChroma', 'Isophonics'), MadMomPCPChromaExtractorShim(samplerate, block_size, step_size), chromas, chord_types, type_templates, self_probs=chord_self_probs, hmm_decoders=hmm_decoders)
# run_experiments('Lists/IsophonicsChords2010.lst', '/import/c4dm-datasets/C4DM Music Collection', '../Ground-Truth/Chord Annotations C4DM', join('Experiments', 'HPCPChroma', 'Isophonics'), MadMomHPCPChromaExtractorShim(samplerate, block_size, step_size), chromas, chord_types, type_templates, self_probs=chord_self_probs, hmm_decoders=hmm_decoders)

# RWC-Popular
# run_experiments('Lists/RWC-Popular.lst', '../Datasets/RWC-Popular', '../Ground-Truth/Chord Annotations MARL/RWC_Pop_Chords', join('Experiments', 'DeepChroma', 'RWC-Popular'), MadMomDeepChromaExtractorShim(samplerate, block_size, step_size), chromas, chord_types, type_templates, audio_suffix='.aiff', self_probs=chord_self_probs, hmm_decoders=hmm_decoders)
# run_experiments('Lists/RWC-Popular.lst', '../Datasets/RWC-Popular', '../Ground-Truth/Chord Annotations MARL/RWC_Pop_Chords', join('Experiments', 'CLPChroma', 'RWC-Popular'), MadMomCLPChromaExtractorShim(samplerate, block_size, step_size), chromas, chord_types, type_templates, audio_suffix='.aiff', self_probs=chord_self_probs, hmm_decoders=hmm_decoders)
# run_experiments('Lists/RWC-Popular.lst', '../Datasets/RWC-Popular', '../Ground-Truth/Chord Annotations MARL/RWC_Pop_Chords', join('Experiments', 'PCPChroma', 'RWC-Popular'), MadMomPCPChromaExtractorShim(samplerate, block_size, step_size), chromas, chord_types, type_templates, audio_suffix='.aiff', self_probs=chord_self_probs, hmm_decoders=hmm_decoders)
# run_experiments('Lists/RWC-Popular.lst', '../Datasets/RWC-Popular', '../Ground-Truth/Chord Annotations MARL/RWC_Pop_Chords', join('Experiments', 'HPCPChroma', 'RWC-Popular'), MadMomHPCPChromaExtractorShim(samplerate, block_size, step_size), chromas, chord_types, type_templates, audio_suffix='.aiff', self_probs=chord_self_probs, hmm_decoders=hmm_decoders)

# Isophonics stereo augmented
# run_experiments('Lists/IsophonicsChords2010-augmentedstereo.lst', '/import/c4dm-01/StereoSeparation', '../Ground-Truth/Chord Annotations C4DM', join('Experiments', 'DeepChroma', 'IsophonicsAugmented'), MadMomDeepChromaExtractorShim(samplerate, block_size, step_size), chromas, chord_types, type_templates, self_probs=chord_self_probs, hmm_decoders=hmm_decoders)
# run_experiments('../Lists/IsophonicsChords2010-augmentedstereo.lst', '/import/c4dm-01/StereoSeparation', '../Ground-Truth/Chord Annotations C4DM', join('Experiments', 'CLPChroma', 'IsophonicsAugmented'), MadMomCLPChromaExtractorShim(samplerate, block_size, step_size), chromas, chord_types, type_templates, self_probs=chord_self_probs, hmm_decoders=hmm_decoders)
