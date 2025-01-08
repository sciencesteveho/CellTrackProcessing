#! /usr/bin/env python
# -*- coding: utf-8 -*-

#: TO-DO:
# - [ ] integrate chymograph code
# - [ ] add fxn to get individual plots
# - [ ] implement proper transforms for merged data
# - [ ] use custom colors pls no default or arthur will yell
# - [ ] add peak selection for merge, such that plots will only include graphs
#   where both transgenes hit the required number of peaks
# - [ ] automatically choose nums for subplots based on parser input


"""Tools for the processing and analysis of cell fluorescence tracks. 

CellTrackProcessor was a simple tool designed to get results as fast as possible
while still being relatively reusable, so please don't interpret some of the
design choices used here as best practice."""


import argparse
import contextlib
import itertools
import os
from typing import Any, Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import more_itertools as mit  # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import scipy.fft
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from skimage import img_as_bool

from utils import list_from_dictvals


class TrackProcessor:
    """Object to process transgene fluorescence tracks from
    fiji/elephant/mastodon.

    Args:
        trackfile_1: name of trackfile
        trackfile_2: name of trackfile
        transgene_1: name of transgene
        transgene_2: name of transgene
        peak_detection: bool - option to run peak detection
        num_peaks_filter: number of peaks required
        fourier transform: bool - option to run fourier transform
        periodicity: bool - option to run periodicity

    # Helpers
        KEEP_COLS -- name of columns to keep in dataframe

    # NOTES 'Spot track ID' = Track ID 'Spot frame' = Frames 'Spot intensity:
    total ch1'
    """

    KEEP_COLS = ["TRACK_ID", "POSITION_X", "POSITION_Y", "FRAME"]

    def __init__(
        self,
        trackfile_1: str,
        trackfile_2: str,
        transgene_1: str,
        transgene_2: str,
        peak_detection: bool,
        num_peaks_filter: int,
        fourier_transform: bool,
        period: bool,
        # plots_per_image: int,
    ):
        """Initialize the class"""
        self.trackfile_1 = trackfile_1
        self.trackfile_2 = trackfile_2
        self.transgene_1 = transgene_1
        self.transgene_2 = transgene_2
        self.peak_detection = peak_detection
        self.num_peaks_filter = num_peaks_filter
        self.fourier_transform = fourier_transform
        self.period = period
        # self.plots_per_image = plots_per_image

        self._set_matplotlib_params()  # set matplotlib plotting style
        self._make_directories()  # make output dir

        # set output filename
        if self.trackfile_1 and self.trackfile_2:
            self.filename = f"{self.transgene_1}_{self.transgene_2}_"
        elif self.period:
            self.filename = f"{self.transgene_1}_periodicity_"
        elif self.fourier_transform:
            self.filename = f"{self.transgene_1}_fourier_"
        else:
            self.filename = f"{self.transgene_1}_"

        # start setting attributes
        self.track_df = self._dict_of_frame_intensities(self.trackfile_1, self.transgene_1)

        # merge if 2 files, else run the gamut
        if self.trackfile_1 and self.trackfile_2:
            track_2 = self._dict_of_frame_intensities(self.trackfile_2, self.transgene_2)
            self.merged = self._combine_tracks(self.track_df, track_2)
        else:
            # get peaks
            self.peaks = (
                self._detect_peaks(self.track_df) if self.peak_detection else {}
            )
            # filter
            if self.num_peaks_filter:
                self.track_df, self.peaks = self._filter_n_peaks(
                    self.track_df, self.peaks
                )

            # fft
            self.fourier = self._fft(self.track_df)

            # calculate periodicity
            self.periodicity = self._periodicity(self.track_df, self.transgene_1)

    def _make_directories(self) -> None:
        """Create output directory"""
        with contextlib.suppress(FileExistsError):
            os.makedirs("../output")

    def _set_matplotlib_params(self) -> None:
        """Set matplotlib parameters"""
        plt.rcParams.update({"font.size": 7})  # set font size
        plt.rcParams["font.family"] = "Helvetica"  # set font
        plt.rcParams["figure.figsize"] = [34, 18]  # set fig size

    def _dict_of_frame_intensities(
        self, trackfile: str, gene: str
    ) -> Dict[int, pd.DataFrame]:
        """Opens CSV and stores each sequence of intensities to its
        corresponding trackID in an individual dataframe.

        Returns:
            intensities -- dictionary with k : v: trackid : dataframe
            each df has a column w/ frame number and max intensity
        """
        df = pd.read_csv(
            trackfile,
            header=[0],
        )

        intensity_key = "TOTAL_INTENSITY_CH1"
        if intensity_key in df.columns:
            self.intensity_key = intensity_key
        else:
            self.intensity_key = "TOTAL_INTENSITY"
        self.KEEP_COLS.append(self.intensity_key)

        track_df = {}
        for idx in df.TRACK_ID.unique():
            sub_df = df[df["TRACK_ID"] == idx]
            sub_df = sub_df[self.KEEP_COLS]
            df_name = f"{gene}_{sub_df['TRACK_ID'].iloc[0]}"
            sub_df.rename(columns={self.intensity_key: df_name}, inplace=True)
            sub_df.name = df_name
            track_df[idx] = sub_df

        return track_df

    def _fft(self, intensities: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """Fourier transform of intensities."""
        df2 = {}
        for item in intensities:
            df2[item] = intensities[item].copy()
            df2[item].iloc[:, -1:] = scipy.fft.fft(df2[item].iloc[:, -1:])
        return df2

    def _detect_peaks(
        self, intensities: Dict[int, pd.DataFrame]
    ) -> Dict[int, pd.DataFrame]:
        """Detect peaks using scipy find_peaks

        Args:
            intensities: dict of pd.dfs with fluorescent intensities

        Returns:
            peak_dict -- dictionary with indexes of peaks for each df
        """
        peak_dict = {}
        for track in intensities.values():
            vals = track.iloc[:, -1:].to_numpy()
            # peaks,_ = find_peaks(vals)
            peaks = argrelextrema(vals, comparator=np.greater, order=2)[0]
            peak_dict[track.name] = peaks
        return peak_dict

    def _chunklist(self, input: List[Any]) -> List[List[Any]]:
        """Chunks a list into groups of 100, returns a list of lists."""
        return [list(c) for c in mit.divide(len(input): 100 + 1, input)]

    def _combine_tracks(
        self, dict1: Dict[int, pd.DataFrame], dict2: Dict[int, pd.DataFrame]
    ) -> Dict[int, pd.DataFrame]:
        """Combine both dicts, merging on frame

        Args:
            dict1: first track dict to merge
            dict2: second track dict to merge

        Returns:
            merged_dict -- combined dictionary
        """
        merged_dict = {}
        for ids in range(len(dict1) - 1):
            merged = pd.merge(
                left=dict1[ids],
                right=dict2[ids],
                how="left",
                left_on="TRACK_ID",
                right_on="TRACK_ID",
            )
            merged_dict[ids] = merged
        return merged_dict

    def _filter_n_peaks(
        self,
        intensities: Dict[int, pd.DataFrame],
        peaks: Dict[str, np.ndarray],
    ) -> Dict[int, pd.DataFrame]:
        """Remove graphs with less than n peaks."""
        trackids_keep = [
            int(item[0].split("_")[1])
            for item in peaks.items()
            if len(item[1]) >= self.num_peaks_filter
        ]
        filtered_intensity = {
            key: value for key, value in intensities.items() if key in trackids_keep
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
        """Calculate periodicity of peaks."""
        periodicity = {}
        for idx, items in enumerate(self.peaks.items()):
            periodicity_keeper = []
            frameid = int(items[0].split("_")[1])
            frames = intensities[frameid]["FRAME"].to_numpy()
            for item in items[1]:
                if item == items[1][0]:
                    periodicity_keeper.append(int(frames[item]) - int(frames[0]))
                else:
                    idx = np.where(items[1] == item)[0][0]
                    periodicity_keeper.append(
                        int(frames[item]) - int(frames[items[1][idx - 1]])
                    )
            periodicity[frameid] = pd.DataFrame(
                periodicity_keeper, columns=[f"{gene}_{frameid}"]
            )

    def _plot_intensities(
        self,
        nrow: int,
        ncol: int,
        frames: List[pd.DataFrame],
        num: int,
        peaks: Dict[str, np.ndarray],
    ) -> None:
        """Plots in matplotlib and saves figure. Chunks into groups of 100, and
        adds dummy frames if len(frames) is less than 100, otherwise will
        throw IndexError.
        """
        if len(frames) < 100:
            difference = 100 - len(frames)
            frames.extend(pd.DataFrame([0], columns=["dummy"]) for _ in range(difference))
        _, axes = plt.subplots(nrow, ncol)
        for count, (r, c) in enumerate(itertools.product(range(nrow), range(ncol))):
            if len(peaks) <= 1:
                frames[count].plot(ax=axes[r, c], color=["blue", "red"])
            elif frames[count].columns[0] == "dummy":
                frames[count].plot(ax=axes[r, c], color=["blue", "red"])
            else:
                linevals = frames[count].iloc[:, 1]
                vals = frames[count].iloc[:, 1:].to_numpy().flatten()
                colname = frames[count].columns[1]
                try:
                    markers = vals[peaks[colname]]
                except IndexError:
                    print(count)
                ax = axes[r, c]
                ax.plot(frames[count].FRAME, linevals)
                with contextlib.suppress(ValueError):
                    ax.plot(savgol_filter(linevals, 15, 3))
                patch = mpatches.Patch(label=colname)
                ax.legend(handles=[patch])
                ax.plot(peaks[colname], markers, "x")
        plt.savefig(
            f"../output/{self.filename}{num}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plotdata(self, intensities: Dict[int, pd.DataFrame]) -> None:
        """Plot data using matplotlib and save to output directory."""
        if self.period:
            peaks = {}
            dfs = list_from_dictvals(self.periodicity)
        elif self.fourier_transform:
            peaks = {}
            dfs = list_from_dictvals(self.fourier)
        else:
            peaks = self.peaks
            dfs = list_from_dictvals(intensities)
            dfs = [frame.iloc[:, -2:] for frame in dfs]

        # plot
        chunks = self._chunklist(dfs)
        for index, miniframe in enumerate(chunks):
            self._plot_intensities(
                nrow=10,
                ncol=10,
                frames=miniframe,
                num=index,
                peaks=peaks,
            )


def main() -> None:
    """Pipeline to process track data"""
    parser = argparse.ArgumentParser(description="Intensity Plots")
    parser.add_argument(
        "-a",
        "--a",
        help="first transgene trackfile.csv from Fiji",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--b",
        help="second transgene trackfile.csv from Fiji",
        type=str,
    )
    parser.add_argument(
        "-an",
        "--a_name",
        help="Name of first transgene",
        type=str,
    )
    parser.add_argument(
        "-bn",
        "--b_name",
        help="Name of first transgene",
        type=str,
    )
    parser.add_argument(
        "-pd",
        "--peak_detection",
        help="Option to run peak detection",
        action="store_true",
    )
    parser.add_argument(
        "-np",
        "--num_peaks_filter",
        help="Option to filter peaks. Use '0' for no filtering",
        type=int,
    )
    parser.add_argument(
        "-fft",
        "--fourier_transform",
        help="Option to run use fourier transform",
        action="store_true",
    )
    parser.add_argument(
        "-pe",
        "--period",
        help="Plot periodicity instead of intensity",
        action="store_true",
    )
    # parser.add_argument('-ppi', '--plots_per_image', help="Number of plots per image", type=int)
    args = parser.parse_args()

    # instantiate object
    trackObject = TrackProcessor(
        args.a,
        args.b,
        args.a_name,
        args.b_name,
        args.peak_detection,
        args.num_peaks_filter,
        args.fourier_transform,
        args.period,
        # args.plots_per_image,
    )

    # run pipeline!
    trackObject.plotdata(intensities=trackObject.track_df)


if __name__ == "__main__":
    main()

"""
import pandas as pd

df = pd.read_csv(
    'test-vertices.csv',
    header=[0,1],
    )
"""
