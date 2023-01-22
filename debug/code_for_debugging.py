#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] try detecta
# - [ ] use custom colors pls no default or arthur will yell
# - [ ] add peak selection for merge, such that plots will only include graphs where both transgenes hit the required number of peaks
# - [ ] automatically choose nums for subplots based on parser input


"""Peak selection and processing for cell fluorescence tracks"""


import csv
import argparse
import numpy as np
import pandas as pd
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import more_itertools as mit

# from detecta import detect_peaks
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
import scipy.fft


def list_from_dictvals(input: Dict[Any, Any]) -> List[Any]:
    return [
        input[key]
        for key in input.keys()
        ]


class FijiTrackProcessor:
    """Object to process transgene fluorescence tracks from fiji/elephant/mastodon

    Args:
        a // name of trackfile
        b // name of trackfile
        a_name // name of transgene
        b_name // name of transgene
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
        a: str,
        b: str, 
        a_name: str, 
        b_name: str,
        peak_detection: bool,
        num_peaks_filter: int,
        fourier_transform: bool,
        periodicity: bool,
        # plots_per_image: int,
        ):
        """Initialize the class"""
        self.a = a
        self.b = b
        self.a_name = a_name
        self.b_name = b_name
        self.peak_detection = peak_detection
        self.num_peaks_filter = num_peaks_filter
        self.fourier_transform = fourier_transform
        self.periodicity = periodicity
        # self.plots_per_image = plots_per_image

        self._set_matplotlib_params()  # set matplotlib plotting style

        # set output filename
        if self.a and self.b:
            self.filename = f'{self.a_name}_{self.b_name}_'
        elif self.periodicity:
            self.filename = f'{self.a_name}_periodicity_'
        elif self.fourier_transform:
            self.filename = f'{self.a_name}_fourier_'
        else:
            self.filename = f'{self.a_name}_'
        
        self._make_directories()  # make output dir

    def _make_directories(self) -> None:
        try:
            os.makedirs('../output')
        except FileExistsError:
            pass
        
    def _set_matplotlib_params(self) -> None:
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
        # def get_idxs(col_fiji, col_mastodon):
        #     try:
        #         return colnames.index(col_fiji)
        #     except ValueError:
        #         return colnames.index(col_mastodon)
            
        frame_idx = colnames.index('FRAME') 
        trackid_idx = colnames.index('TRACK_ID')
        try:
            intensity_idx = colnames.index('TOTAL_INTENSITY_CH1')
        except ValueError:
            intensity_idx = colnames.index('TOTAL_INTENSITY')

        with open(trackfile, newline = '') as file:
            file_reader = csv.reader(file, delimiter=',')
            next(file_reader)  # skip header
            intensities = {}
            for index, items in enumerate(file_reader):
                if index == 0:  # first trackid
                    trackid = items[trackid_idx]
                    templist = []
                    templist.append(
                        (items[frame_idx],
                        int(items[intensity_idx]))
                        )
                elif index != length-2:  # append if not a new id, else, continue
                    if items[trackid_idx] != str(int(trackid) + 1):
                        templist.append(
                            (items[frame_idx],
                            int(items[intensity_idx]))
                            )
                    else:
                        intensities[int(trackid)] = pd.DataFrame(
                            templist,
                            columns=['frame', gene + "_" + trackid]
                            )
                        templist = []
                        trackid = items[trackid_idx]
                        templist.append(
                            (items[frame_idx],
                            int(items[intensity_idx]))
                            )
                else:  # append at last line
                    templist.append(
                        (items[frame_idx],
                        int(items[intensity_idx]))
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
                if self.periodicity or len(self.peaks) <= 1:
                    frames[count].plot(ax=axes[r,c], color=['blue', 'red'])
                else: 
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

        plt.savefig((f'../output/{self.filename}{str(num)}.png'), format='png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_oscillations(self) -> None:
        """_summary_

        Args:
            a // _description_
            b // _description_

        Raises:
            AssertionError: _description_
        
        Returns:
            c -- _description_
        """
        self.trackfile = self._dict_of_frame_intensities(self.a, self.a_name)

        # get peaks
        if self.peak_detection:
            self.peaks = self._detect_peaks(self.trackfile)
        else:
            self.peaks = {}

        # filter
        if self.num_peaks_filter:
            self.trackfile, self.peaks = self._filter_n_peaks(self.trackfile, self.peaks)

        # calculate periodicity
        if self.periodicity:
            self.trackfile = self._periodicity(self.trackfile, self.a_name)

        # track merge if two files are provided
        if self.a and self.b:
            track_2 = self._dict_of_frame_intensities(self.b, self.b_name)
            merged = self._combine_tracks(self.trackfile, track_2)
            self.trackfile = list_from_dictvals(merged)
        else:
            self.trackfile = list_from_dictvals(self.trackfile)
        
        # plot 
        chunks = self._chunklist(self.trackfile)
        if self.fourier_transform:
            for chunk in chunks:
                for subchunk in chunk:
                    subchunk.iloc[:,1] = scipy.fft.fft(subchunk.iloc[:,1])
        for index, miniframe in enumerate(chunks):
            self._plot_intensities(
                10,
                10,
                miniframe,
                index,
                )


def main() -> None:
    # instantiate object
    fijitrackObject = FijiTrackProcessor(
        '../../single_cells.csv',
        None,
        'her1',
        None,
        True,
        2,
        False,
        False,
        # args.plots_per_image,
        )
    
    # # run pipeline! 
    # fijitrackObject.plot_oscillations()

    fijitrackObject.trackfile = fijitrackObject._dict_of_frame_intensities(fijitrackObject.a, fijitrackObject.a_name)

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
        fijitrackObject.trackfile = fijitrackObject._periodicity(fijitrackObject.trackfile, fijitrackObject.a_name)

    # track merge if two files are provided
    if fijitrackObject.a and fijitrackObject.b:
        track_2 = fijitrackObject._dict_of_frame_intensities(fijitrackObject.b, fijitrackObject.b_name)
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
        
test = fijitrackObject.trackfile[996]
track = test
vals = track.iloc[:,1:]
vals = vals.to_numpy().flatten()
# peaks,_ = find_peaks(vals)
peaks = argrelextrema(
    vals,
    comparator=np.greater,
    order=2
    )