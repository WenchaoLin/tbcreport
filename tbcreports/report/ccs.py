
"""
Generate a report summarizing Circular Consensus Read (CCS) results.
"""

from collections import OrderedDict
import functools
import os
import sys
import logging
import argparse
import time
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np

from pbcommand.models.report import (Report, Table, Column, Attribute, Plot,
                                     PlotGroup)
from pbcommand.models import FileTypes, get_pbparser
from pbcommand.cli import pbparser_runner
from pbcommand.utils import setup_log
from pbcore.io import ConsensusReadSet

from pbreports.plot.helper import (get_fig_axes_lpr, apply_histogram_data,
                                   get_blue, get_green, Line, apply_line_data)
from pbreports.util import accuracy_as_phred_qv

log = logging.getLogger(__name__)
__version__ = '0.44'


class Constants(object):

    """ ids for the core Report objects (e.g., Plot, PlotGroup, etc...)"""
    TOOL_ID = "pbreports.tasks.ccs_report"
    TOOL_NAME = "ccs_report"
    DRIVER_EXE = "python -m pbreports.report.ccs --resolved-tool-contract"

    R_ID = "ccs"

    # PlotGroup
    PG_READLENGTH = 'readlength_group'
    PG_ACCURACY = "accuracy_group"
    PG_NPASSES = "npasses_hist"

    # Plots
    P_READLENGTH = "readlength_hist"
    P_ACCURACY = "accuracy_hist"
    P_NPASSES = "npasses_hist"
    P_SCATTER = "npasses_vs_accuracy"

    I_CCS_READ_LENGTH_HIST = "ccs_readlength_hist.png"
    I_CCS_READ_ACCURACY_HIST = "ccs_accuracy_hist.png"
    I_CCS_NUM_PASSES_HIST = "ccs_npasses_hist.png"
    I_CCS_SCATTER_PLOT = "ccs_npasses_vs_accuracy.png"

    # Attributes
    A_NREADS = 'number_of_ccs_reads'
    A_TOTAL_BASES = 'total_number_of_ccs_bases'
    A_MEAN_READLENGTH = 'mean_ccs_readlength'
    A_MEAN_ACCURACY = 'mean_accuracy'
    #A_MEAN_QV = 'mean_qv'
    A_MEAN_NPASSES = 'mean_ccs_num_passes'

    # Table
    T_ID = 'ccs_table'

    # Columns
    C_MOVIE_NAME = 'movie_name'
    C_NBASES = "number_of_ccs_bases"
    C_TOTAL_BASES = 'total_number_of_ccs_bases'
    C_MEAN_READLENGTH = 'ave_ccs_readlength'
    C_MEAN_ACCURACY = 'ave_ccs_accuracy'
    #C_MEAN_QV = 'mean_ccs_qv'
    C_MEAN_NPASSES = 'mean_ccs_num_passes'

    ATTR_LABELS = OrderedDict([
        (A_NREADS, "CCS reads"),
        (A_TOTAL_BASES, "Number of CCS bases"),
        (A_MEAN_READLENGTH, "CCS Read Length (mean)"),
        (A_MEAN_ACCURACY, "CCS Read Score (mean)"),
        #(A_MEAN_QV, "Mean Consensus Predicted QV"),
        (A_MEAN_NPASSES, "Number of Passes (mean)")
    ])
    ATTR_DESCRIPTIONS = {
        A_NREADS: "The number of CCS reads",
        A_TOTAL_BASES: "Total number of consensus bases in all CCS reads",
        A_MEAN_READLENGTH: "Mean length of CCS reads",
        A_MEAN_ACCURACY: "Mean predicted accuracy of CCS reads",
        #A_MEAN_QV: "Phred log-scale QV, equivalent to mean accuracy",
        A_MEAN_NPASSES: "Mean number of complete subreads per CCS read, rounded to the nearest integer"
    }


class MovieResult(object):

    """Simple container class to hold the results of Movie (bax)"""

    def __init__(self, file_name, movie_name, read_lengths, accuracies, num_passes):
        self.file_name = file_name
        self.movie_name = movie_name
        # these are all np.array
        self.read_lengths = read_lengths
        self.accuracies = accuracies
        self.num_passes = num_passes

    def __str__(self):
        _d = dict(k=self.__class__.__name__,
                  m=self.movie_name,
                  f=os.path.basename(self.file_name))
        return "{k} {m} {f}".format(**_d)

    def __repr__(self):
        return "<" + str(self) + " > "


# FIXME(nechols)(2016-02-18): this is very inefficient
def _bam_file_to_movie_results(file_name):
    """
    Read what is assumed to be a single BAM file (as a ConsensusReadSet).
    """
    from pbcore.io import IndexedBamReader
    results = []
    with IndexedBamReader(file_name) as bam:
        for rg in bam.readGroupTable:
            assert rg["ReadType"] == "CCS"

        movies = list(set([rg["MovieName"] for rg in bam.readGroupTable]))
        for movie_name in movies:
            def _base_calls():
                for r in bam:
                    if r.movieName == movie_name:
                        yield r.peer.query_length

            def _num_passes():
                for r in bam:
                    if r.movieName == movie_name:
                        yield r.numPasses

            def _accuracy():
                for r in bam:
                    if r.movieName == movie_name:
                        yield r.readScore

            read_lengths = np.fromiter(_base_calls(), dtype=np.int64, count=-1)
            num_passes = np.fromiter(_num_passes(), dtype=np.int64, count=-1)
            accuracy = np.fromiter(_accuracy(), dtype=np.float, count=-1)

            results.append(MovieResult(
                file_name, movie_name, read_lengths, accuracy, num_passes))
        return results


def _movie_results_to_attributes(movie_results):
    """Create the necessary attributes for the CCS report"""
    rs = [m.read_lengths for m in movie_results]
    read_lengths = np.concatenate(rs)
    ac = [m.accuracies for m in movie_results]
    accuracies = np.concatenate(ac)
    npass = [m.num_passes for m in movie_results]
    num_passes = np.concatenate(npass)

    m_readlength = int(read_lengths.mean()) if read_lengths.size > 0 else 0.0
    m_accuracy = np.round(
        accuracies.mean(), decimals=4) if accuracies.size > 0 else 0.0
    m_npasses = np.round(num_passes.mean()) if num_passes.size > 0 else 0.0
    #m_qv = int(round(accuracy_as_phred_qv(float(m_accuracy))))

    n_reads_at = Attribute(
        Constants.A_NREADS, read_lengths.shape[0], name=Constants.ATTR_LABELS[
            Constants.A_NREADS])
    t_bases_at = Attribute(
        Constants.A_TOTAL_BASES, read_lengths.sum(), name=Constants.ATTR_LABELS[Constants.A_TOTAL_BASES])
    m_readlength_at = Attribute(
        Constants.A_MEAN_READLENGTH, m_readlength, name=Constants.ATTR_LABELS[Constants.A_MEAN_READLENGTH])
    m_accuracy_at = Attribute(
        Constants.A_MEAN_ACCURACY, m_accuracy, name=Constants.ATTR_LABELS[Constants.A_MEAN_ACCURACY])
    #m_qv = Attribute(
    #    Constants.A_MEAN_QV, m_qv, name=Constants.ATTR_LABELS[Constants.A_MEAN_QV])
    m_npasses_at = Attribute(
        Constants.A_MEAN_NPASSES, m_npasses, name=Constants.ATTR_LABELS[Constants.A_MEAN_NPASSES])

    attributes = [n_reads_at, t_bases_at,
                  m_readlength_at, m_accuracy_at, m_npasses_at]

    return attributes


def _movie_results_to_table(movie_results):
    """Group movie results by movie name and build a report table.

    Table has movie name, # of CCS bases, Total CCS bases,
    mean CCS readlength and mean CCS accuracy.
    """
    labels = Constants.ATTR_LABELS
    columns = [Column(Constants.C_MOVIE_NAME, header="Movie"),
               Column(Constants.A_NREADS, header=labels[Constants.A_NREADS]),
               Column(Constants.A_TOTAL_BASES,
                      header=labels[Constants.A_TOTAL_BASES]),
               Column(Constants.A_MEAN_READLENGTH,
                      header=labels[Constants.A_MEAN_READLENGTH]),
               Column(Constants.A_MEAN_ACCURACY,
                      header=labels[Constants.A_MEAN_ACCURACY]),
               #Column(Constants.A_MEAN_QV,
               #       header="Mean Consensus Predicted QV"),
               Column(Constants.A_MEAN_NPASSES,
                      header=labels[Constants.A_MEAN_NPASSES])]

    table = Table(Constants.T_ID, title="By Movie", columns=columns)

    movie_names = {m.movie_name for m in movie_results}

    for movie_name in movie_names:
        rs = [
            m.read_lengths for m in movie_results if m.movie_name == movie_name]
        read_lengths = np.concatenate(rs)
        ac = [
            m.accuracies for m in movie_results if m.movie_name == movie_name]
        accuracies = np.concatenate(ac)
        npass = [
            m.num_passes for m in movie_results if m.movie_name == movie_name]
        num_passes = np.concatenate(npass)

        m_readlength = int(
            read_lengths.mean()) if read_lengths.size > 0 else 0.0
        m_accuracy = np.round(
            accuracies.mean(), decimals=4) if accuracies.size > 0 else 0.0
        m_npasses = np.round(
            num_passes.mean(), decimals=3) if num_passes.size > 0 else 0.0
        #m_qv = int(round(accuracy_as_phred_qv(float(accuracies.mean()))))

        table.add_data_by_column_id(Constants.C_MOVIE_NAME, movie_name)
        table.add_data_by_column_id(Constants.A_NREADS, read_lengths.shape[0])
        table.add_data_by_column_id(
            Constants.A_TOTAL_BASES, read_lengths.sum())
        table.add_data_by_column_id(Constants.A_MEAN_READLENGTH, m_readlength)
        table.add_data_by_column_id(Constants.A_MEAN_ACCURACY, m_accuracy)
        #table.add_data_by_column_id(Constants.A_MEAN_QV, m_qv)
        table.add_data_by_column_id(Constants.A_MEAN_NPASSES, m_npasses)

    return table


def _make_histogram(data, axis_labels, nbins, barcolor):
    """Create a fig, ax instance and generate a histogram.

    :param data: np.array
    :param axis_labels: (tuple of str) (axis label, y axis label)
    :return: matplotlib fig, ax
    """
    # axis_labels = ('Median Distance Between Adapters', 'Pre-Filter Reads')
    fig, ax = get_fig_axes_lpr()
    apply_histogram_data(
        ax, data, nbins, axis_labels=axis_labels, barcolor=barcolor)
    return fig, ax


def to_cdf(points):
    _total = 0
    data = []
    for x, y in points:
        _total += int(x * y)
        data.append(_total)
    return data


def _make_histogram_with_cdf(data, axis_labels, nbins, barcolor):
    """

    """
    fig, ax = _make_histogram(data, axis_labels, nbins, barcolor)

    bins, bin_edges = np.histogram(data, bins=nbins)

    rax = ax.twinx()

    log.debug(
        "Min edges {e} bins {b}".format(e=len(bin_edges), b=len(bins)))

    cdf = to_cdf(zip(bin_edges[:-1], bins))
    max_cdf = max(cdf)
    sdf = [max_cdf - i for i in cdf]

    log.debug((len(bin_edges), len(sdf)))

    # Plot the data
    rax.plot(bin_edges[:-1], sdf, 'k')
    rax.set_xlim(bin_edges.min(), bin_edges.max())

    if len(axis_labels) == 3:
        rax.set_ylabel(axis_labels[2])

    return fig, ax


def _custom_histogram_with_cdf(new_rlabel, threshold, data, axis_labels, nbins, barcolor):
    fig, ax = _make_histogram(data, axis_labels, nbins, barcolor)

    bins, bin_edges = np.histogram(data, bins=nbins)

    rax = ax.twinx()

    log.debug(
        "Min edges {e} bins {b}".format(e=len(bin_edges), b=len(bins)))

    cdf = to_cdf(zip(bin_edges[:-1], bins))
    max_cdf = max(cdf)

    exceeded_threshold = False
    if max_cdf > threshold:
        exceeded_threshold = True
        tmp_cdf = [x / float(threshold) for x in cdf]
        cdf = tmp_cdf
        max_cdf = max(cdf)

    sdf = [max_cdf - i for i in cdf]

    log.debug((len(bin_edges), len(sdf)))

    # Plot the data
    rax.plot(bin_edges[:-1], sdf, 'k')
    rax.set_xlim(bin_edges.min(), bin_edges.max())

    if len(axis_labels) == 3:
        if exceeded_threshold:
            rax.set_ylabel(new_rlabel)
        else:
            # use the default rlabel given
            rax.set_ylabel(axis_labels[2])

    return fig, ax


def scatter_plot_accuracy_vs_numpasses(data,
                                       axis_labels=(
                                           "Number of passes",
                                           "Read Score as Phred QV"),
                                       nbins=None, barcolor=None):
    """
    """
    npasses, accuracy = data
    qvs = accuracy_as_phred_qv(accuracy)
    fig, ax = get_fig_axes_lpr()
    data = [Line(xData=npasses,
                 yData=qvs,
                 style='o')]
    apply_line_data(
        ax=ax,
        line_models=data,
        axis_labels=axis_labels,
        only_whole_ticks=False)
    return fig, ax


def create_plot(_make_plot_func, plot_id, axis_labels, nbins, plot_name, barcolor, data, output_dir, dpi=72):
    """Internal function used to create Plot instances.

    This should probably have a special container class to capture all the
    plot config options.
    """

    fig, ax = _make_plot_func(data, axis_labels, nbins, barcolor)
    path = os.path.join(output_dir, plot_name)
    try:
        fig.tight_layout()
    except AttributeError as e:  # FIXME bug 25872
        log.warn("figure.tight_layout() not available")
        log.warn(str(e))
    except ValueError as e:
        log.error(str(e))
    fig.savefig(path, dpi=dpi)
    log.debug("Saved plot with id {i} to {p}".format(p=path, i=plot_id))
    thumbnail = plot_name.replace(".png", "_thumb.png")

    to_b = lambda x: os.path.basename(x)
    fig.savefig(os.path.join(output_dir, thumbnail), dpi=20)
    plt.close(fig)
    log.debug("Saved plot to {p}".format(p=thumbnail))
    plot = Plot(plot_id, to_b(plot_name), thumbnail=to_b(thumbnail))

    return plot

# These functions create signatures (data, axis_labels, nbins, barcolor
_custom_read_length_histogram = functools.partial(
    _custom_histogram_with_cdf, "Mb > Read Length", 1000000)
_custom_read_accuracy_histogram = functools.partial(
    _custom_histogram_with_cdf, "Mb > Read Score", 1000000)


# These functions need to generate a function with signature (data,
# output_dir, dpi=)
create_readlength_plot = functools.partial(create_plot, _custom_read_length_histogram, Constants.P_READLENGTH,
                                           ("Read Length", "Reads", "bp > Read Length"), 80, Constants.I_CCS_READ_LENGTH_HIST, get_blue(3))

create_accuracy_plot = functools.partial(create_plot, _custom_read_accuracy_histogram, Constants.P_ACCURACY,
                                         ("Quality", "Reads", "bp > Read Score"), 80, Constants.I_CCS_READ_ACCURACY_HIST, get_green(3))

create_npasses_plot = functools.partial(create_plot, _make_histogram, Constants.P_NPASSES,
                                        ("Number of Passes", "Reads"), 80, Constants.I_CCS_NUM_PASSES_HIST, "#F18B17")

create_scatter_plot = functools.partial(create_plot,
                                        scatter_plot_accuracy_vs_numpasses, Constants.P_SCATTER,
                                        ("Number of passes",
                                         "Read Score as Phred QV"), None,
                                        Constants.I_CCS_SCATTER_PLOT, get_blue(3))


def to_report(ccs_subread_set, output_dir):
    bam_files = list(ccs_subread_set.toExternalFiles())
    log.info("Generating report from files: {f}".format(f=bam_files))
    movie_results = []
    for fn in bam_files:
        movie_results.extend(_bam_file_to_movie_results(fn))
    log.debug("\n" + pformat(movie_results))

    rs = [m.read_lengths for m in movie_results]
    readlengths = np.concatenate(rs)
    ac = [m.accuracies for m in movie_results]
    accuracies = np.concatenate(ac)
    ps = [m.num_passes for m in movie_results]
    num_passes = np.concatenate(ps)

    readlength_plot = create_readlength_plot(readlengths, output_dir)
    accuracy_plot = create_accuracy_plot(accuracies, output_dir)
    npasses_plot = create_npasses_plot(num_passes, output_dir)
    scatter_plot = create_scatter_plot((num_passes, accuracies), output_dir)

    readlength_group = PlotGroup(Constants.PG_READLENGTH,
                                 title="CCS Read Length",
                                 plots=[readlength_plot],
                                 thumbnail=readlength_plot.thumbnail)
    accuracy_group = PlotGroup(Constants.PG_ACCURACY, plots=[accuracy_plot],
                               thumbnail=accuracy_plot.thumbnail,
                               title="CCS Read Score")

    npasses_group = PlotGroup(Constants.P_NPASSES, plots=[npasses_plot],
                              thumbnail=npasses_plot.thumbnail,
                              title="Number of Passes")

    scatter_group = PlotGroup(Constants.P_SCATTER, plots=[scatter_plot],
                              thumbnail=scatter_plot.thumbnail,
                              title="Number of Passes vs. Read Score")

    table = _movie_results_to_table(movie_results)
    log.info(str(table))

    attributes = _movie_results_to_attributes(movie_results)

    report = Report(Constants.R_ID, tables=[table], attributes=attributes,
                    plotgroups=[readlength_group, accuracy_group,
                                npasses_group, scatter_group],
                    dataset_uuids=(ccs_subread_set.uuid,))

    return report


def run_report(
        input_file,
        report_json,
        output_dir):
    log.info("Running {f} v{v}.".format(
        f=os.path.basename(__file__), v=__version__))
    report = None
    ds = ConsensusReadSet(input_file)
    report = to_report(ds, output_dir)
    log.info(pformat(report.to_dict()))
    report.write_json(report_json)
    return 0


def args_runner(args):
    return run_report(
        input_file=args.ccs_in,
        report_json=args.report_json,
        output_dir=args.output_dir)


def resolved_tool_contract_runner(rtc):
    return run_report(
        input_file=rtc.task.input_files[0],
        report_json=rtc.task.output_files[0],
        output_dir=os.path.dirname(rtc.task.output_files[0]))


def get_parser():
    p = get_pbparser(
        tool_id=Constants.TOOL_ID,
        version=__version__,
        name=Constants.TOOL_NAME,
        description=__doc__,
        driver_exe=Constants.DRIVER_EXE)
    ap = p.arg_parser.parser
    p.add_input_file_type(FileTypes.DS_CCS, "ccs_in",
                          name="ConsensusReadSet",
                          description="ConsensusRead DataSet file")
    p.add_output_file_type(FileTypes.REPORT, "report_json",
                           name="CCS report",
                           description="Path to write Report json output.",
                           default_name="ccs_report")
    ap.add_argument('-o', '--output-dir', dest='output_dir',
                    default=os.getcwd(),
                    help="Path to write histogram images to.")
    # ap.add_argument('--debug', action='store_true',
    #               help='Flag to debug to stdout.')
    return p


def main(argv=sys.argv):
    """Main point of Entry"""
    return pbparser_runner(
        argv=argv[1:],
        parser=get_parser(),
        args_runner_func=args_runner,
        contract_runner_func=resolved_tool_contract_runner,
        alog=log,
        setup_log_func=setup_log)

if __name__ == "__main__":
    sys.exit(main())
