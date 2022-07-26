# gwcluster

CLI for clustering LIGO's noise sources.

## Installation

```bash
git clone https://github.com/mmuldo/gwcluster.git
cd gwcluster
conda env create -f environment.yml
conda activate gwcluster
```

## Config file

A config file stored at `~/.gwcluster.yml` can be written in order to avoid
specifying parameters on the command-line.

### Quickstart

From the root directory where you installed gwcluster,

```bash
cp gwcluster.yml.example ~/.gwcluster.yml
```

Then edit `~/.gwcluster.yml` in your favorite text editor.

## CLI

All commands must be run from the root directory where you installed gwcluster.

### Help

```bash
python -m gwcluster [subcommand] --help
```

### Seismometer

#### Help

```bash
python -m gwcluster seismometer --help
```

#### Example

```bash
python -m --clustering KMeans --clustering-kwargs 'n_clusters=3,init=kmeans++' gwcluster seismometer --ifo C1 --system PEM --signal BS --start '2022-07-13 00:00:00' --end '2022-07-14 00:00:00' --host $LIGO_HOST --port $LIGO_PORT --output $HOME/code/gwcluster.wiki/summary --cache-data
```

*NB*: In lieu of specifying parameters via the command line, the can be sepcified in `~/.gwcluster.yml`.
