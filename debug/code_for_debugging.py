#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [X] implement fft 
# - [X] calculate periodicity 
#   - [X] plot period on the Y axis, number of cycles on the X axis
# - [ ] automatically get idx cols from CSV header
# - [ ] add peak selection for merge, such that plots will only include graphs where both transgenes hit the required number of peaks
# - [ ] automatically choose nums for subplots based on parser input


"""Peak selection and processing for cell fluorescence tracks"""

import csv
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from typing import Any, Dict, List

import matplotlib.patches as mpatches
import more_itertools as mit
import scipy.fft
from scipy.signal import find_peaks


def list_from_dictvals(input: Dict[Any, Any]) -> List[Any]:
    return [
        input[key]
        for key in input.keys()
        ]


class FijiTrackProcessor:
    """Object to process transgene fluorescence tracks from fiji/elephant/mastodon

    Args:
        trackfile_1 // name of trackfile
        trackfile_2 // name of trackfile
        gene_1 // name of transgene
        gene_2 // name of transgene
        peak_detection // bool - option to run peak detection
        num_peaks_filter // number of peaks required for downstream
        fourier transformer // _summary
        periodicity // _summary

    Methods
    ----------
    _set_matplotlib_params:
        lorem ipsum
    _chunklist:
        lorem ipsum
    _dict_of_frame_intensities:
        lorem ipsum
    _detect_peaks:
        lorem ipsum
    _combine_tracks:
        lorem ipsum
    _filter_n_peaks:
        lorem ipsum
    _plot_intensities:
        lorem ipsum
    plot_oscillations:
        lorem ipsum

    # Helpers
        LOREM -- ipsum
    """

    def __init__(
        self,
        trackfile_1: str,
        trackfile_2: str, 
        gene_1: str, 
        gene_2: str,
        peak_detection: bool,
        num_peaks_filter: int,
        fourier_transform: bool,
        periodicity: bool,
        # plots_per_image: int,
        ):
        """Initialize the class"""
        self.trackfile_1 = trackfile_1
        self.trackfile_2 = trackfile_2
        self.gene_1 = gene_1
        self.gene_2 = gene_2
        self.peak_detection = peak_detection
        self.num_peaks_filter = num_peaks_filter
        self.fourier_transform = fourier_transform
        self.periodicity = periodicity
        # self.plots_per_image = plots_per_image

        # set matplotlib plotting style
        self._set_matplotlib_params()

        # set output filename based on number of provided tracks
        if self.trackfile_1 and self.trackfile_2:
            self.filename = f'{self.gene_1}_{self.gene_2}_'
        elif self.periodicity:
            self.filename = f'{self.gene_1}_periodicity_'
        else:
            self.filename = f'{self.gene_1}_'

    def _set_matplotlib_params(self):
        plt.rcParams.update({'font.size': 7})  # set font size
        plt.rcParams["font.family"] = 'Helvetica'  # set font
        plt.rcParams["figure.figsize"] = [34,18]  # set fig size

    def _chunklist(self, input: List[Any]) -> List[List[Any]]:
        return [
            list(c)
            for c in mit.divide(int(len(input)/100) + 1, input)
            ]

    def _dict_of_frame_intensities(
        self,
        trackfile: str,
        gene: str
        ) -> Dict[int, pd.DataFrame]:
        """Opens CSV and stores each sequence of intensities to corresponding trackID
        in an individual dataframe

        Returns:
            intensities -- dictionary with k : v // trackid : dataframe
        """
        with open(trackfile, newline = '') as file:
            length = len(file.readlines())

        with open(trackfile, newline = '') as file:
            file_reader = csv.reader(file, delimiter=',')
            colnames = []
            for row in file_reader:
                colnames.append(row)
                break
            
        colnames = colnames[0]
        frame_idx = colnames.index('FRAME')
        trackid_idx = colnames.index('TRACK_ID')
        intensity_idx = colnames.index('TOTAL_INTENSITY_CH1')

        with open(trackfile, newline = '') as file:
            file_reader = csv.reader(file, delimiter=',')
            next(file_reader)  # skip header
            intensities = {}
            for index, items in enumerate(file_reader):
                if index == 0:  # first trackid
                    trackid = items[self.trackid_idx]
                    templist = []
                    templist.append(
                        (items[self.frame_idx],
                        int(items[self.intensity_idx]))
                        )
                elif index != length-2:  # append if not a new id, else, continue
                    if items[self.trackid_idx] != str(int(trackid) + 1):
                        templist.append(
                            (items[self.frame_idx],
                            int(items[self.intensity_idx]))
                            )
                    else:
                        intensities[int(trackid)] = pd.DataFrame(
                            templist,
                            columns=['frame', gene + "_" + trackid]
                            )
                        templist = []
                        trackid = items[self.trackid_idx]
                        templist.append(
                            (items[self.frame_idx],
                            int(items[self.intensity_idx]))
                            )
                else:  # append at last line
                    templist.append(
                        (items[self.frame_idx],
                        int(items[self.intensity_idx]))
                        )
                    intensities[int(trackid)] = pd.DataFrame(
                        templist,
                        columns=['frame', gene + "_" + trackid]
                        )
        return intensities
    
    def _detect_peaks(self, intensities: Dict[int, pd.DataFrame]) -> None:
        """Detect peaks using scipy find_peaks
        
        Args:
            intensities // dict of pd.dfs with fluorescent intensities
            
        Returns:
            peak_dict -- dictionary with indexes of peaks for each df
        """
        peak_dict = {}
        for track in intensities.values():
            vals = track.iloc[:,1:]
            vals = vals.to_numpy().flatten()
            peaks,_ = find_peaks(vals)
            peak_dict[track.columns[1]] = peaks
        return peak_dict
    
    def _combine_tracks(
        self,
        dict1: Dict[int, pd.DataFrame],
        dict2: Dict[int, pd.DataFrame]
        ) -> Dict[int, pd.DataFrame]:
        """Combine both dicts, merging on frame

        Args:
            dict1 // first track dict to merge
            dict2 // second track dict to merge

        Returns:
            merged_dict -- combined dictionary
        """
        merged_dict = {}
        for ids in range(0, len(dict1)-1):
            merged = pd.merge(
                left = dict1[ids],
                right=dict2[ids],
                how='left',
                left_on='frame',
                right_on='frame'
                )
            merged_dict[ids] = merged
        return merged_dict
    
    def _filter_n_peaks(
        self,
        intensities: Dict[int, pd.DataFrame],
        peaks: Dict[str, np.ndarray],
        ) -> Dict[int, pd.DataFrame]:
        """Remove graphs with less than n peaks"""
        trackids_keep = [
            int(item[0].split("_")[1])
            for item in peaks.items()
            if len(item[1]) >= self.num_peaks_filter
            ]
        filtered_intensity = {
            key: value
            for key, value in intensities.items()
            if key in trackids_keep
            }
        filtered_peaks = {
            key: value 
            for key, value in peaks.items()
            if int(key.split("_")[1]) in trackids_keep
            }
        return filtered_intensity, filtered_peaks
    
    def _periodicity(
        self,
        intensities: Dict[int, pd.DataFrame],
        gene: str,
        ) -> None:
        """_summary_"""
        periodicity = {}
        for idx, items in enumerate(self.peaks.items()):
            periodicity_keeper = []
            frameid = int(items[0].split("_")[1])
            frames = intensities[frameid].iloc[:,0]
            for item in items[1]:
                if item == items[1][0]:
                    periodicity_keeper.append(int(frames[item]) - int(frames[0]))
                else:
                    idx = np.where(items[1] == item)[0][0]
                    periodicity_keeper.append(int(frames[item]) - int(frames[items[1][idx-1]]))
            periodicity[frameid] = pd.DataFrame(periodicity_keeper, columns=[gene + "_" + str(frameid)])
        return periodicity

    def _plot_intensities(
        self,
        nrow,
        ncol,
        frames,
        num,
        ):
        """Plots in matplotlib and saves figure. Chunks into groups of 100, and adds
        dummy frames if len(frames) is less than 100, otherwise will throw IndexError.
        """
        if len(frames) < 100:
            difference = 100 - len(frames)
            for i in range(0, difference):
                frames.append(pd.DataFrame([0], columns=['dummy']))

        count=0
        _, axes = plt.subplots(nrow, ncol)
        for r in range(nrow):
            for c in range(ncol):
                if self.periodicity:
                    frames[count].plot(ax=axes[r,c], color=['blue', 'red'])
                elif len(self.peaks) > 1:
                    if frames[count].columns[0] == 'dummy':
                        frames[count].plot(ax=axes[r,c], color=['blue', 'red'])
                    else:
                        linevals = frames[count].iloc[:,1]
                        vals = frames[count].iloc[:,1:].to_numpy().flatten()
                        colname = frames[count].columns[1]
                        try:
                            markers = vals[self.peaks[colname]]
                        except IndexError:
                            print(count)
                        ax=axes[r,c]
                        ax.plot(frames[count].index, linevals)
                        patch = mpatches.Patch(label=colname)
                        ax.legend(handles=[patch])
                        ax.plot(self.peaks[colname], markers, "x")
                # else:
                #     frames[count].plot(ax=axes[r,c], color=['blue', 'red'])
                count += 1

        plt.savefig((f'{self.filename}{str(num)}.png'), format='png', dpi=300, bbox_inches='tight')
        plt.close()


def main() -> None:
    """Pipeline to process track data"""
    parser = argparse.ArgumentParser(description='Intensity Plots')
    parser.add_argument('-track_1', '--trackfile_1', help='first transgene trackfile.csv from Fiji', type=str)
    parser.add_argument('-track_2', '--trackfile_2', help='second transgene trackfile.csv from Fiji', type=str)
    parser.add_argument('-g1', '--gene_1', help='Name of first transgene', type=str)
    parser.add_argument('-g2', '--gene_2', help='Name of first transgene', type=str)
    parser.add_argument('-f', '--frame_idx', help='0-indexed column number for "FRAME"', type=int)
    parser.add_argument('-i', '--intensity_idx', help='0-indexed column number for "TOTAL_INTENSITY_CH1"', type=int)
    parser.add_argument('-t', '--trackid_idx', help='0-indexed column number for "TRACK_ID"', type=int)
    parser.add_argument('-p', '--peak_detection', help="Option to run peak detection", action='store_true')
    parser.add_argument('-n', '--num_peaks_filter', help="Option to filter peaks. Use '0' for no filtering", type=int)
    parser.add_argument('-fft', '--fourier_transform', help="Option to run use fourier transform", action='store_true')
    parser.add_argument('-peri', '--periodicity', help="Plot periodicity instead of intensity", action='store_true')
    # parser.add_argument('-ppi', '--plots_per_image', help="Number of plots per image", type=int)
    args = parser.parse_args()

    # instantiate object
    fijitrackObject = FijiTrackProcessor(
        '../single_cells.csv',
        None,
        'her1',
        None,
        8,
        15,
        2,
        True,
        2,
        False,
        True,
        # args.plots_per_image,
        )
    
    fijitrackObject.trackfile = fijitrackObject._dict_of_frame_intensities(fijitrackObject.trackfile_1, fijitrackObject.gene_1)

    # get peaks
    if fijitrackObject.peak_detection:
        fijitrackObject.peaks = fijitrackObject._detect_peaks(fijitrackObject.trackfile)
    else:
        fijitrackObject.peaks = {}

    # filter
    if fijitrackObject.num_peaks_filter:
        fijitrackObject.trackfile, fijitrackObject.peaks = fijitrackObject._filter_n_peaks(fijitrackObject.trackfile, fijitrackObject.peaks)

    # calculate periodicity
    if fijitrackObject.periodicity:
        fijitrackObject.trackfile = fijitrackObject._periodicity(fijitrackObject.trackfile, fijitrackObject.gene_1)

    # track merge if two files are provided
    if fijitrackObject.trackfile_1 and fijitrackObject.trackfile_2:
        track_2 = fijitrackObject._dict_of_frame_intensities(fijitrackObject.trackfile_2, fijitrackObject.gene_2)
        merged = fijitrackObject._combine_tracks(fijitrackObject.trackfile, track_2)
        fijitrackObject.trackfile = list_from_dictvals(merged)
    else:
        fijitrackObject.trackfile = list_from_dictvals(fijitrackObject.trackfile)
    
    # plot 
    chunks = fijitrackObject._chunklist(fijitrackObject.trackfile)
    if fijitrackObject.fourier_transform:
        for chunk in chunks:
            for subchunk in chunk:
                subchunk.iloc[:,1] = scipy.fft.fft(subchunk.iloc[:,1])
    for index, miniframe in enumerate(chunks):
        fijitrackObject._plot_intensities(
            10,
            10,
            miniframe,
            index,
            )

'''
python fiji_track_processor.py \
    --trackfile_1 ../single_cells.csv \
    --gene_1 her1 

python fiji_track_processor.py \
    --trackfile_1 ../single_cells.csv \
    --gene_1 her1 \
    --peak_detection \
    --num_peaks_filter 2

python fiji_track_processor.py \
    --trackfile_1 ../single_cells.csv \
    --gene_1 her1 \
    --peak_detection \
    --num_peaks_filter 2 \
    --periodicity

python fiji_track_processor.py \
    --trackfile_1 ../single_cells.csv \
    --gene_1 her1 \
    --fourier_transform

python fiji_track_processor.py \
    --trackfile_1 ../her1.csv \
    --trackfile_2 ../securin.csv \
    --gene_1 her1 \
    --gene_2 securin 


'''