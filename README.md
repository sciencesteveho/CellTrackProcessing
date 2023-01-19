# FijiProcessing
Tools to plot and analyze oscillations from fiji tracks.
&nbsp;
<div align="center">
    <img src='images/plot_example.png'>
</div>
&nbsp;

## Usage
Note: CSVs must have intensity columns titled either "TOTAL_INTENSITY_CH1" or "TOTAL_INTENSITY"

```sh
# plot a single trackfile
$ python fiji_track_processor.py \
    --trackfile_1 ../single_cells.csv \
    --gene_1 her1 
```

```sh
# plot a single track file w/ peak detection and a minimum of 2 peaks per dataset
$ python fiji_track_processor.py \
    --trackfile_1 ../single_cells.csv \
    --gene_1 her1 \
    --peak_detection \
    --num_peaks_filter 2
```

```sh
# plot periodicity (frames between peaks)
$ python fiji_track_processor.py \
    --trackfile_1 ../single_cells.csv \
    --gene_1 her1 \
    --peak_detection \
    --num_peaks_filter 2 \
    --periodicity
```

```sh
# plot fourier transformed graphs (frequency)
$ python fiji_track_processor.py \
    --trackfile_1 ../single_cells.csv \
    --gene_1 her1 \
    --fourier_transform
```

```sh
# plot two trackfiles merged
$ python fiji_track_processor.py \
    --trackfile_1 ../her1.csv \
    --trackfile_2 ../securin.csv \
    --gene_1 her1 \
    --gene_2 securin 
```


## Dependencies

```sh
$ pip install csv numpy pandas more_itertools matplotlib scipy
```





