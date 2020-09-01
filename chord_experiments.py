#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from os.path import join, basename, splitext, isfile
from chord_estimation import HMMSmoothedChordsFromTemplates, run_on_file_list, run_on_file_list_with_arg, evaluate_chords, MadMomDeepChromaExtractor, MadMomCLPChromaExtractor, MadMomPCPChromaExtractor, MadMomHPCPChromaExtractor, RmsEnergyExtractor, LibrosaCENSChromaExtractor
import pandas as pd
import pickle
import os
from os.path import dirname
import time

def run_experiments(list_path, audio_dir, reference_label_dir, output_dir, chroma_extractor, chromas, chord_types, type_templates, self_prob, audio_suffix='.wav', chordfile_suffix='.lab', hmm_decoders=['decodeMAP'], entropy_experiments=False, energies=False):
    list_name = splitext(basename(list_path))[0]
    if not all([isfile(join(output_dir, 'HMMChords-{}-{}Ps'.format(y, round(self_prob,3)), '{}-ResultsMirex.txt'.format(list_name))) for y in hmm_decoders]):

        chromagramStartTime = time.time()
        # Calculate chromagrams
        chromagrams = run_on_file_list(list_path, lambda x: chroma_extractor(join(audio_dir, x+audio_suffix)), verbose=False)
        chromagramEndTime = time.time()
        print('Time spent calculating chromagrams:', chromagramEndTime-chromagramStartTime)

        # Save chromagrams
        if 'Isophonics' in output_dir:
            dataset_name = 'Isophonics'
        elif 'RWC-Popular' in output_dir:
            dataset_name = 'RWC-Popular'

        if 'DeepChroma' in output_dir:
            chroma_type = 'DeepChroma'
        elif 'CLPChroma' in output_dir:
            chroma_type = 'CLPChroma'
        elif 'CENSChroma' in output_dir:
            chroma_type = 'CENSChroma'
        chromagram_numpy_list = [np.asarray(chromagram[0]) for chromagram in chromagrams] #weird but needed to convert CLPChroma/CENSChroma to numpy array
        chromagram_file = join('Experiments', chroma_type, dataset_name, 'Chromagrams')
        try:
            os.makedirs(dirname(chromagram_file))
        except OSError:
            pass
        print('Saving Chromagrams at' + chromagram_file)
        np.savez(join(chromagram_file), chromagram_numpy_list)

        # Calculate RMS energies of chromagrams
        if energies:
            energy_extractor = RmsEnergyExtractor(block_size, step_size, samplerate)
            rms_energies = run_on_file_list(list_path, lambda x: energy_extractor(join(audio_dir, x+audio_suffix)), verbose=False)

        # Calculate chords with confidence from chromagrams
        HMMStartTime = time.time()
        for decoder in hmm_decoders:
            chord_dir = join(output_dir, 'HMMChords-{}-{}Ps'.format(decoder, round(self_prob,3)))
            if entropy_experiments:
                entropies = run_on_file_list_with_arg(list_path, HMMSmoothedChordsFromTemplates(audio_dir, chord_dir, chromas, chord_types, type_templates, chroma_extractor, self_prob, audio_suffix, chordfile_suffix, decode_method=decoder, is_entropy=True), chromagrams, verbose=False)
                if entropies:
                    pd.DataFrame(entropies).to_csv(join(chord_dir, list_name+'-entropies.csv'), index=False, header=False)
                if energies:
                    pd.DataFrame(rms_energies).to_csv(join(chord_dir, list_name+'-rms_energies.csv'))
            else:
                log_probs_and_confidences = run_on_file_list_with_arg(list_path, HMMSmoothedChordsFromTemplates(audio_dir, chord_dir, chromas, chord_types, type_templates, chroma_extractor, self_prob, audio_suffix, chordfile_suffix, decode_method=decoder), chromagrams, verbose=False)
                if log_probs_and_confidences:
                    pd.DataFrame(log_probs_and_confidences).to_csv(join(chord_dir, list_name+'-logprobs_confidences.csv'), index=False, header=False)
        HMMEndTime = time.time()
        print('Time spent decoding HMMs:', HMMEndTime-HMMStartTime)

    # Evaluate chord sequences
    for decoder in hmm_decoders:
        hmm_scores = []
        chord_dir = join(output_dir, 'HMMChords-{}-{}Ps'.format(decoder, round(self_prob, 3)))
        hmm_scores.append(evaluate_chords(list_path, reference_label_dir, chord_dir, '4TriadsOutput', join(chord_dir, '{}-ResultsMirex.txt'.format(list_name)), overwrite=True))
        print('{} {} HMM chord scores: {}'.format(list_name, decoder, hmm_scores))


## Chroma extraction configuration
samplerate = 44100
block_size = 8192
# step_size = 4410
step_size = 4096

## Chord estimation configuration
type_templates = np.array([[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]])
chord_types = ['maj', 'min', 'dim', 'aug']
chromas = ['A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab']
chord_self_probs = 0.1
hmm_decoders = ['decodeMAP_with_medianOPC', 'decode_with_PPD']
entropy_hmm_decoders = ['decodeMAP_with_sequential_entropy', 'decodeMAP_with_framewise_entropy']


####### Experiments #######


print(' Starting running experiments..')

# PPD experiments
# Isophonics
# run_experiments('Lists/IsophonicsChords2010.lst', 'Audio/Songs.nosync', 'Ground-Truth', join('Experiments', 'DeepChroma', 'Isophonics'), MadMomDeepChromaExtractor(samplerate, block_size, 4410), chromas, chord_types, type_templates, self_prob=chord_self_probs, hmm_decoders=hmm_decoders)
# run_experiments('Lists/IsophonicsChords2010.lst', 'Audio/Songs.nosync', 'Ground-Truth', join('Experiments', 'CLPChroma', 'Isophonics'), MadMomCLPChromaExtractor(samplerate, block_size, step_size), chromas, chord_types, type_templates, self_prob=chord_self_probs, hmm_decoders=hmm_decoders)
# run_experiments('Lists/IsophonicsChords2010.lst', 'Audio/Songs.nosync', 'Ground-Truth', join('Experiments', 'CENSChroma', 'Isophonics'), LibrosaCENSChromaExtractor(samplerate, block_size, step_size), chromas, chord_types, type_templates, self_prob=chord_self_probs, hmm_decoders=hmm_decoders)

# RWC-Popular
# run_experiments('Lists/RWC-Popular.lst', 'Audio/Songs.nosync/Popular_Music', 'Ground-Truth/RWC-Popular', join('Experiments', 'DeepChroma', 'RWC-Popular'), MadMomDeepChromaExtractor(samplerate, block_size, 4410), chromas, chord_types, type_templates, audio_suffix='.aiff', self_prob=chord_self_probs, hmm_decoders=hmm_decoders)
# run_experiments('Lists/RWC-Popular.lst', 'Audio/Songs.nosync/Popular_Music', 'Ground-Truth/RWC-Popular', join('Experiments', 'CLPChroma', 'RWC-Popular'), MadMomCLPChromaExtractor(samplerate, block_size, step_size), chromas, chord_types, type_templates, audio_suffix='.aiff', self_prob=chord_self_probs, hmm_decoders=hmm_decoders)
# run_experiments('Lists/RWC-Popular.lst', 'Audio/Songs.nosync/Popular_Music', 'Ground-Truth/RWC-Popular', join('Experiments', 'CENSChroma', 'RWC-Popular'), LibrosaCENSChromaExtractor(samplerate, block_size, step_size), chromas, chord_types, type_templates, audio_suffix='.aiff', self_prob=chord_self_probs, hmm_decoders=hmm_decoders)


# Observation Entropy Experiments
# Isophonics
run_experiments('Lists/IsophonicsChords2010.lst', 'Audio/Songs.nosync', 'Ground-Truth', join('Experiments', 'DeepChroma', 'Isophonics'), MadMomDeepChromaExtractor(samplerate, block_size, 4410), chromas, chord_types, type_templates, self_prob=chord_self_probs, hmm_decoders=entropy_hmm_decoders, entropy_experiments=True, energies=True)
run_experiments('Lists/IsophonicsChords2010.lst', 'Audio/Songs.nosync', 'Ground-Truth', join('Experiments', 'CLPChroma', 'Isophonics'), MadMomCLPChromaExtractor(samplerate, block_size, step_size), chromas, chord_types, type_templates, self_prob=chord_self_probs, hmm_decoders=entropy_hmm_decoders, entropy_experiments=True, energies=True)
run_experiments('Lists/IsophonicsChords2010.lst', 'Audio/Songs.nosync', 'Ground-Truth', join('Experiments', 'CENSChroma', 'Isophonics'), LibrosaCENSChromaExtractor(samplerate, block_size, step_size), chromas, chord_types, type_templates, self_prob=chord_self_probs, hmm_decoders=entropy_hmm_decoders, entropy_experiments=True, energies=True)

# RWC-Popular
run_experiments('Lists/RWC-Popular.lst', 'Audio/Songs.nosync/Popular_Music', 'Ground-Truth/RWC-Popular', join('Experiments', 'DeepChroma', 'RWC-Popular'), MadMomDeepChromaExtractor(samplerate, block_size, 4410), chromas, chord_types, type_templates, audio_suffix='.aiff', self_prob=chord_self_probs, hmm_decoders=entropy_hmm_decoders, entropy_experiments=True)
run_experiments('Lists/RWC-Popular.lst', 'Audio/Songs.nosync/Popular_Music', 'Ground-Truth/RWC-Popular', join('Experiments', 'CLPChroma', 'RWC-Popular'), MadMomCLPChromaExtractor(samplerate, block_size, step_size), chromas, chord_types, type_templates, audio_suffix='.aiff', self_prob=chord_self_probs, hmm_decoders=entropy_hmm_decoders, entropy_experiments=True)
run_experiments('Lists/RWC-Popular.lst', 'Audio/Songs.nosync/Popular_Music', 'Ground-Truth/RWC-Popular', join('Experiments', 'CENSChroma', 'RWC-Popular'), LibrosaCENSChromaExtractor(samplerate, block_size, step_size), chromas, chord_types, type_templates, audio_suffix='.aiff', self_prob=chord_self_probs, hmm_decoders=entropy_hmm_decoders, entropy_experiments=True)



print('Completed Experiments.')
