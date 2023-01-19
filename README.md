# FijiProcessing
Tools to plot and analyze oscillations from fiji tracks.

&nbsp;

## Usage

```sh
# plot a single trackfile
$ python fiji_track_processor.py \
    --trackfile_1 path/to/single_cells.csv \
    --gene_1 her1 \
    --frame_idx 8 \
    --intensity_idx 15 \
    --trackid_idx 2 
```

```sh
# plot a single track file w/ peak detection and a minimum of 2 peaks per dataset
$ python fiji_track_processor.py \
    --trackfile_1 path/to/single_cells.csv \
    --gene_1 her1 \
    --frame_idx 8 \
    --intensity_idx 15 \
    --trackid_idx 2 \
    --peak_detection \
    --num_peaks_filter 2
```

Note: if using multiple CSVs, you MUST format the CSVs to have the same column indexes for "FRAME", "TOTAL_INTENSITY_CH1", and "TRACK_ID"
```sh
# plot two trackfiles merged
$ python fiji_track_processor.py \
    --trackfile_1 path/to/her1.csv \
    --gene_1 her1 \
    --trackfile_2 path/to/securin.csv \
    --gene_2 securin \
    --frame_idx 8 \
    --intensity_idx 26 \
    --trackid_idx 2
```
&nbsp;
## Dependencies

```sh
$ pip install csv numpy pandas more_itertools matplotlib scipy
```





