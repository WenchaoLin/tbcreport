#!/usr/bin/env python

from collections import OrderedDict
import functools
import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from pbcommand.models.report import Table, Column, Report, Plot, PlotGroup
from pbcommand.validators import validate_dir, validate_file
from pbcommand.cli.core import pacbio_args_runner
from pbcommand.utils import setup_log
from pbcore.io.FastqIO import FastqReader

from pbreports.plot.helper import get_fig_axes_lpr

log = logging.getLogger(__name__)
__version__ = '1.2'


class FastqStats(object):

    def __init__(self, reads, qvs, file_name):
        """Simple container class"""
        self.qvs = qvs
        # these are read lengths
        self.reads = reads
        self.file_name = file_name

    @staticmethod
    def from_file(file_name):
        qvs, reads = _get_stats(file_name)
        return FastqStats(reads, qvs, file_name)

    def __str__(self):
        outs = list()
        outs.append("Reads           :{n}".format(n=self.reads.shape[0]))
        outs.append("Mean readlength :{m}".format(int(np.sum(self.reads))))
        outs.append("Total bases     :{m}".format(m=int(np.sum(self.reads))))
        outs.append("Mean qv         :{m:.2f}".format(m=self.qvs.mean()))
        return "\n".join(outs)


def _get_stats(fastq_file_name):
    raw_qvs = np.array([r.quality for r in FastqReader(fastq_file_name)])
    qvs = np.hstack(raw_qvs)
    reads = np.array([len(r.sequence) for r in FastqReader(fastq_file_name)])
    return qvs, reads


def _generate_histogram(datum, title, xlabel, ylabel=None):
    fig, ax = get_fig_axes_lpr()
    fig.suptitle(title)
    ax.hist(datum)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return fig, ax


def __generate_histogram_comparison(method_name, title, xlabel, list_fastq_stats):
    fig, ax = get_fig_axes_lpr()
    fig.suptitle(title)

    alpha = 0.3
    hs = OrderedDict()
    for fastq_stat in list_fastq_stats:
        label = os.path.basename(fastq_stat.file_name)
        h = ax.hist(getattr(fastq_stat, method_name),
                    alpha=alpha, bins=85, label=label)
        hs[label] = h

    ax.set_xlabel(xlabel)
    ax.legend(loc="best")
    return fig, ax

to_qv_histogram = functools.partial(
    __generate_histogram_comparison, 'qvs', "Quality Values", "Quality Values")
to_read_length_histogram = functools.partial(
    __generate_histogram_comparison, 'reads', "Read Length", "Read Length")


def _generate_table(list_fastq_stats):
    columns = [Column('file_name', header='File Name'),
               Column('n_reads', header="Number of Reads"),
               Column('total_bases', header="Total Bases"),
               Column('mean_readlength', header="Mean Read Length"),
               Column('mean_qv', header="Mean Quality Values")]

    table = Table('fastq_table', columns=columns)

    for fastq_stat in list_fastq_stats:
        table.add_data_by_column_id(
            'file_name', os.path.basename(fastq_stat.file_name))
        table.add_data_by_column_id('n_reads', fastq_stat.reads.shape[0])
        table.add_data_by_column_id(
            'total_bases', int(np.sum(fastq_stat.reads)))
        table.add_data_by_column_id(
            'mean_readlength', int(fastq_stat.reads.mean()))
        table.add_data_by_column_id('mean_qv', np.round(
            fastq_stat.qvs.mean(), decimals=2))

    return table


def fastq_files_to_stats(fastq_files):
    fastq_stats = {file_name: FastqStats.from_file(
        file_name) for file_name in fastq_files}
    return fastq_stats


def to_report(fastq_files, qv_hist=None, readlength_hist=None):
    """Generate a histogram of read lengths and quality values"""
    fastq_stats = fastq_files_to_stats(fastq_files)

    table = _generate_table(fastq_stats.values())
    log.info(str(table))

    if qv_hist is not None:
        fig, ax = to_qv_histogram(fastq_stats.values())
        fig.savefig(qv_hist)
    if readlength_hist is not None:
        fig, ax = to_read_length_histogram(fastq_stats.values())
        fig.savefig(readlength_hist)
    plt.close(fig)
    readlength_hist_plot = Plot('readlength_hist', readlength_hist)
    plotgroup = PlotGroup('readlength_group', title="Read Length Histogram", plots=[
                          readlength_hist_plot])
    report = Report('ccs_validator', tables=[table], plotgroups=[plotgroup])
    return report


def args_runner(args):
    output_dir = args.output_dir
    json_report_name = args.report

    to_p = lambda x: os.path.join(output_dir, x)
    json_report = to_p(json_report_name)
    readlength_hist = to_p(
        'ccs_validation_readlength_histogram.png') if output_dir else None
    qv_hist = to_p('ccs_validation_qv_histogram.png') if output_dir else None

    log.info("Starting v{v} of {f}".format(v=__version__,
                                           f=os.path.basename(__file__)))
    fastq_files = [args.fastq_1, args.fastq_2]

    # weak attempt to make the plots labels show up consistently
    fastq_files.sort()
    report = to_report(fastq_files, qv_hist=qv_hist,
                       readlength_hist=readlength_hist)

    log.info("writing report to {j}".format(j=json_report))
    report.write_json(json_report)

    return 0


def get_parser():
    p = argparse.ArgumentParser(version=__version__)
    p.add_argument('fastq_1', type=validate_file)
    p.add_argument('fastq_2', type=validate_file)
    p.add_argument('--output-dir', required=True, type=validate_dir,
                   dest='output_dir',
                   help="Directory to write Read length and Quality Value histograms.")
    p.add_argument('-r', '--report', type=str,
                   default="ccs_validator_report.json",
                   help="Name of Json report file.")
    p.add_argument('--debug', action='store_true', help='Debug to stdout.')
    return p


def main(argv=sys.argv):
    """Main point of Entry"""
    log.info("Starting {f} version {v} report generation".format(
        f=__file__, v=__version__))
    return pacbio_args_runner(
        argv=argv[1:],
        parser=get_parser(),
        args_runner_func=args_runner,
        alog=log,
        setup_log_func=setup_log)


if __name__ == '__main__':
    sys.exit(main())
