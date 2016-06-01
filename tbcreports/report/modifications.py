# -*- coding: utf-8 -*-
# Created on Tue Jul 31 09:15:42 2012
#
#@author: pmarks

# FIXME no way is this the correct docstring for this report!
"""
Generates a table showing consensus stats and a report showing variants plots
for the top 25 contigs of the supplied reference.
"""

import collections
import argparse
import logging
import gzip
import csv
import os
import sys

from pylab import legend, arange
import numpy as np

from pbcommand.models.report import Report, PlotGroup, Plot
from pbcommand.models import TaskTypes, FileTypes, get_pbparser
from pbcommand.cli import pbparser_runner
from pbcommand.common_options import add_debug_option
from pbcommand.utils import setup_log

import pbreports.plot.helper as PH
from pbreports.util import (add_base_and_plot_options,
                            add_base_options_pbcommand)
from pbreports.util import Constants as BaseConstants

log = logging.getLogger(__name__)

__version__ = '2.1'


class Constants(BaseConstants):
    TOOL_ID = "pbreports.tasks.modifications_report"
    DRIVER_EXE = "python -m pbreports.report.modifications --resolved-tool-contract"


def _create_fig_template(dims=(8, 6), facecolor='#ffffff', gridcolor='#e0e0e0'):
    fig, ax = PH.get_fig_axes_lpr(dims=dims)
    ax = fig.add_subplot(111)

    ax.axesPatch.set_facecolor(facecolor)
    ax.grid(color=gridcolor, linewidth=0.5, linestyle='-')
    ax.set_axisbelow(True)
    PH.set_tick_label_font_size(ax, 12, 12)
    PH.set_axis_label_font_size(ax, 16)
    return fig, ax


def readModificationCsvGz(fn):

    def _open_file(file_name):
        if file_name.endswith(".gz"):
            return gzip.GzipFile(file_name)
        else:
            return open(file_name, "r")

    with _open_file(fn) as f:
        reader = csv.reader(f)

        records = []
        header = reader.next()

        colIdx = 0
        colMap = {}
        for h in header:
            colMap[h] = colIdx
            colIdx += 1

        # Read csv
        n = 0
        kinHit = collections.namedtuple("kinHit", "base coverage score")
        for row, record in enumerate(reader):
            if int(record[colMap['score']]) > 20:
                tupleRec = kinHit(base=record[colMap['base']], coverage=int(
                    record[colMap['coverage']]), score=int(record[colMap['score']]))
                records.append(tupleRec)
                n += 1

        # convert to recarray
        kinRec = [('base', '|S1'), ('coverage', '>i4'),
                  ('score', '>i4'), ('color', 'b')]
        kinArr = np.zeros(len(records), dtype=kinRec)
        idx = 0
        for rec in records:
            kinArr['base'][idx] = rec.base
            kinArr['coverage'][idx] = rec.coverage
            kinArr['score'][idx] = rec.score
            idx += 1

        return kinArr


def plot_kinetics_scatter(kinArr, ax):

    handles = []
    colors = ['red', 'green', 'blue', 'magenta']
    bases = ['A', 'C', 'G', 'T']

    for base, color in zip(bases, colors):
        baseHits = kinArr[kinArr['base'] == base]

        if baseHits.shape[0] > 0:
            # Add a bit of scatter to avoid ugly aliasing in plot due to
            # integer quantization
            cov = baseHits['coverage'] + 0.25 * \
                np.random.randn(baseHits.shape[0])
            score = baseHits['score'] + 0.25 * \
                np.random.randn(baseHits.shape[0])

            pl = ax.scatter(cov, score, c=color, label=base,
                            lw=0, alpha=0.3, s=12)
            handles.append(pl)

    ax.set_xlabel('Per-Strand Coverage')
    ax.set_ylabel('Modification QV')
    legend(handles, bases, loc='upper left')

    if kinArr.shape[0] > 0:
        ax.set_xlim(0, np.percentile(kinArr['coverage'], 95.0) * 1.4)
        ax.set_ylim(0, np.percentile(kinArr['score'], 99.9) * 1.3)


def plot_kinetics_hist(kinArr, ax):

    colors = ['red', 'green', 'blue', 'magenta']
    bases = ['A', 'C', 'G', 'T']

    # Check for empty or peculiar modifications report:
    d = kinArr['score']
    if d.size == 0:
        binLim = 1.0
    elif np.isnan(np.sum(d)):
        binLim = np.nanmax(d)
    else:
        binLim = np.percentile(d, 99.9) * 1.2

    ax.set_xlim(0, binLim)
    bins = arange(0, binLim, step=binLim / 75)

    for base, color in zip(bases, colors):
        baseHits = kinArr[kinArr['base'] == base]
        if baseHits.shape[0] > 0:
            pl = ax.hist(baseHits['score'], color=color,
                         label=base, bins=bins, histtype="step", log=True)

    ax.set_ylabel('Bases')
    ax.set_xlabel('Modification QV')

    if d.size > 0:
        ax.legend(loc='upper right')


def get_qmod_plot(kinData, output_dir, dpi):
    """
    Return a plot object
    """
    fig, ax = _create_fig_template()

    plot_kinetics_scatter(kinData, ax)

    png_path = os.path.join(output_dir, "kinetic_detections.png")
    png, thumbpng = PH.save_figure_with_thumbnail(fig, png_path, dpi=dpi)

    return Plot('kinetic_detections', os.path.basename(png),
                thumbnail=os.path.basename(thumbpng))


def get_qmod_hist(kinData, output_dir, dpi):
    """
    Return a plot object
    """
    fig, ax = _create_fig_template()

    plot_kinetics_hist(kinData, ax)

    png_path = os.path.join(output_dir, "kinetic_histogram.png")
    png, thumbpng = PH.save_figure_with_thumbnail(fig, png_path, dpi=dpi)

    return Plot('kinetic_histogram', os.path.basename(png),
                thumbnail=os.path.basename(thumbpng))


def make_modifications_report(modifications_csv, report, output_dir, dpi=72, dumpdata=True):
    """
    Entry point to report generation.
    """

    kinData = readModificationCsvGz(modifications_csv)

    scatter = get_qmod_plot(kinData, output_dir, dpi)
    hist = get_qmod_hist(kinData, output_dir, dpi)

    pg = PlotGroup('kinetic_detections',
                   title='Kinetic Detections',
                   thumbnail=scatter.thumbnail,
                   plots=[scatter, hist])

    rpt = Report('modifications', plotgroups=[pg])
    rpt.write_json(os.path.join(output_dir, report))


#-----------------------------------------------------------------------
# FIXME DEPRECATED (still used in pbreports tool)
def _args_runner(args):
    make_modifications_report(args.csv, args.report,
                              args.output, args.dpi, args.dumpdata)
    return 0


def add_options_to_parser(p):
    from pbreports.io.validators import validate_file
    p.description = __doc__  # FIXME which is probably wrong
    p.version = __version__
    p = add_base_and_plot_options(p)
    p.add_argument("csv", help="modifications.csv.gz", type=validate_file)
    p.set_defaults(func=_args_runner)
    return p

#-----------------------------------------------------------------------
# TOOL CONTRACT INTERFACE


def args_runner(args):
    return make_modifications_report(
        modifications_csv=args.csv,
        report=os.path.basename(args.report),
        output_dir=args.output)


def resolved_tool_contract_runner(resolved_tool_contract):
    rtc = resolved_tool_contract
    return make_modifications_report(
        modifications_csv=rtc.task.input_files[0],
        report=os.path.basename(rtc.task.output_files[0]),
        output_dir=os.path.dirname(rtc.task.output_files[0]))


def get_parser():
    p = get_pbparser(
        Constants.TOOL_ID,
        __version__,
        "Modifications Report",
        __doc__,
        Constants.DRIVER_EXE,
        is_distributed=True)
    p.add_input_file_type(FileTypes.CSV, "csv", "CSV file",
                          "CSV file of base modifications")
    add_base_options_pbcommand(p, "Basemods report")
    return p


def main(argv=sys.argv):
    mp = get_parser()
    return pbparser_runner(argv[1:],
                           mp,
                           args_runner,
                           resolved_tool_contract_runner,
                           log,
                           setup_log)


if __name__ == "__main__":
    sys.exit(main())
