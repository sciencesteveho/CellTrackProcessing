#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] calculate periodicity 
# - [ ] add peak selection for merge, such that plots will only include graphs where both transgenes hit the required number of peaks
# - [ ] automatically choose nums for subplots based on parser input


"""Peak selection and processing for cell fluorescence tracks"""

import csv
import argparse
import numpy as np
import pandas as pd

import more_itertools as mit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from typing import Any, Dict, List
from scipy.signal import find_peaks


def list_from_dictvals(input: Dict[Any, Any]) -> List[Any]:
    return [
        input[key]
        for key in input.keys()
        ]


class FijiTrackProcessor:
    """Object to process transgene fluorescence tracks from fiji/elephant/mastodon

        args.trackfile_1,
        args.trackfile_2,
        args.gene_1,
        args.gene_2,
        args.frame_idx,
        args.intensity_idx,
        args.trackid_idx,
        args.peak_detection,
        args.num_peaks_filter,

    Args:
        trackfile_1 // name of trackfile
        trackfile_2 // name of trackfile
        gene_1 // name of transgene
        gene_2 // name of transgene
        frame_idx // column index
        intensity_idx // column index
        trackid_idx // column index
        peak_detection // bool - option to run peak detection
        num_peaks_filter // number of peaks required for downstream

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
        frame_idx: int,
        intensity_idx: int,
        trackid_idx: int,
        peak_detection: bool,
        num_peaks_filter: int,
        # plots_per_image: int,
        ):
        """Initialize the class"""
        self.trackfile_1 = trackfile_1
        self.trackfile_2 = trackfile_2
        self.gene_1 = gene_1
        self.gene_2 = gene_2
        self.frame_idx = frame_idx
        self.intensity_idx = intensity_idx
        self.trackid_idx = trackid_idx
        self.peak_detection = peak_detection
        self.num_peaks_filter = num_peaks_filter
        # self.plots_per_image = plots_per_image

        # set matplotlib plotting style
        self._set_matplotlib_params()

        # set output filename based on number of provided tracks
        if self.trackfile_1 and self.trackfile_2:
            self.filename = f'{self.gene_1}_{self.gene_2}_'
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
        return {
            key: value
            for key, value in intensities.items()
            if key in trackids_keep
            }

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
                if len(self.peaks) > 1:
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
                else:
                    frames[count].plot(ax=axes[r,c], color=['blue', 'red'])
                count += 1

        plt.savefig((f'{self.filename}{str(num)}.png'), format='png', dpi=300, bbox_inches='tight')
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
        self.trackfile = self._dict_of_frame_intensities(self.trackfile_1, self.gene_1)

        # get peaks
        if self.peak_detection:
            self.peaks = self._detect_peaks(self.trackfile)
        else:
            self.peaks = {}

        # track merge if two files are provided
        if self.trackfile_1 and self.trackfile_2:
            track_2 = self._dict_of_frame_intensities(self.trackfile_2, self.gene_2)
            merged = self._combine_tracks(self.trackfile, track_2)
            self.trackfile = list_from_dictvals(merged)
        else:
            self.trackfile = list_from_dictvals(self.trackfile)
        
        # plot
        chunks = self._chunklist(self.trackfile)
        for index, miniframe in enumerate(chunks):
            self._plot_intensities(
                10,
                10,
                miniframe,
                index,
                )


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
    # parser.add_argument('-ppi', '--plots_per_image', help="Number of plots per image", type=int)
    args = parser.parse_args()

    # instantiate object
    fijitrackObject = FijiTrackProcessor(
        args.trackfile_1,
        args.trackfile_2,
        args.gene_1,
        args.gene_2,
        args.frame_idx,
        args.intensity_idx,
        args.trackid_idx,
        args.peak_detection,
        args.num_peaks_filter,
        # args.plots_per_image,
        )
    
    # run pipeline! 
    fijitrackObject.plot_oscillations()


if __name__ == '__main__':
    main()

'''
# If using multiple CSVs, you MUST format the CSVs to have the same column index for "FRAME", "TOTAL_INTENSITY_CH1", and "TRACK_ID"

# plot a single trackfile
python fiji_track_processor.py \
    --trackfile_1 ../single_cells.csv \
    --gene_1 her1 \
    --frame_idx 8 \
    --intensity_idx 15 \
    --trackid_idx 2 \
    --peak_detection \
    --num_peaks_filter 2

# plot two trackfiles merged
python fiji_track_processor.py \
    --trackfile_1 ../her1.csv \
    --gene_1 her1 \
    --trackfile_2 ../securin.csv \
    --gene_2 securin \
    --frame_idx 8 \
    --intensity_idx 26 \
    --trackid_idx 2
'''