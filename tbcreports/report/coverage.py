#!/usr/bin/env python

"""
Generates a report showing coverage plots for the top 25 contigs of the
supplied reference.
"""

from collections import OrderedDict
import argparse
import hashlib
import logging
import re
import os.path as op
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from pbcommand.models.report import Attribute, Report, PlotGroup, Plot, PbReportError
from pbcommand.models import FileTypes, get_pbparser
from pbcommand.cli import pbparser_runner
from pbcommand.utils import setup_log
from pbcore.io import GffReader, ReferenceSet

from pbreports.io.validators import validate_file, validate_dir
from pbreports.util import get_top_contigs, add_base_and_plot_options
from pbreports.plot.helper import (get_fig_axes_lpr, apply_line_data,
                                   apply_line_fill_data, apply_histogram_data,
                                   LineFill, save_figure_with_thumbnail)


log = logging.getLogger(__name__)

__version__ = '0.1'


class Constants(object):
    TOOL_ID = "pbreports.tasks.coverage_report"
    DRIVER_EXE = "python -m pbreports.report.coverage --resolved-tool-contract "
    MAX_CONTIGS_ID = "pbreports.task_options.max_contigs"
    MAX_CONTIGS_DEFAULT = 25

    COLOR_STEEL_BLUE_DARK = '#226F96'
    COLOR_STEEL_BLUE_LIGHT = '#2B8CBE'

    A_COVERAGE = "depth_coverage_mean"
    A_MISSING = "missing_bases_pct"

    ATTR_LABELS = OrderedDict([
        (A_COVERAGE, "Mean Coverage"),
        (A_MISSING, "Missing Bases (%)")
    ])

    ATTR_DESCRIPTIONS = {
        A_COVERAGE: "Mean depth of coverage over entire reference",
        A_MISSING: "Percent of reference bases without coverage"
    }


def _create_coverage_plot_grp(top_contigs, cov_map, output_dir):
    """
    Returns io.model.PlotGroup object
    Create the plotGroup element that contains the coverage plots of the top contigs.
    :param top_contigs: (list of Contig objects) sorted by contig size
    :param cov_map: (dict string:ContigCoverage) mapping of contig.id to ContigCoverage object
    :param output_dir: (string) where to write images
    """
    plots = []
    thumbnail = None
    idx = 0
    log.debug('Creating plots for {n} top contig(s)'.format(
        n=str(len(top_contigs))))
    for tc in top_contigs:
        if not tc.id in cov_map:
            # no coverage of this contig
            log.debug('contig {c} has no coverage info '.format(c=tc.id))
            continue
        ctg_cov = cov_map[tc.id]
        fig, ax = _create_contig_plot(ctg_cov)

        fname = os.path.join(output_dir, ctg_cov.file_name)
        if thumbnail is None:
            imgfiles = save_figure_with_thumbnail(fig, fname)
            thumbnail = os.path.basename(imgfiles[1])
        else:
            fig.savefig(fname)
        plt.close(fig)
        id_ = 'coverage_contig_{i}'.format(i=str(idx))
        caption = "Observed depth of coverage across {c} (window size = {b}bp)."
        plot = Plot(id_, os.path.basename(fname), caption.format(
            c=ctg_cov.name, b=ctg_cov.aveRegionSize()))
        plots.append(plot)
        idx += 1

    plot_group = PlotGroup('coverage_plots', title='Coverage Across Reference',
                           thumbnail=thumbnail, plots=plots)
    return plot_group


def _create_coverage_histo_plot_grp(stats, output_dir):
    """
    Returns io.model.PlotGroup object
    Create the plotGroup element that contains the coverage plot histogram
    :param stats: (ReferenceStats) see _get_reference_coverage_stats
    :param output_dir: (string) where to write images
    """
    fig, ax = _create_histogram(stats)
    fname, thumb = [os.path.basename(f) for f in save_figure_with_thumbnail(
        fig, os.path.join(output_dir, 'coverage_histogram.png'))]
    plot = Plot('coverage_histogram', fname, 'Depth of coverage distribution ')
    plot_group = PlotGroup('coverage_histogram_plot_group',
                           title='Depth of Coverage',
                           thumbnail=thumb, plots=[plot])
    return plot_group


def _validate_inputs(gff, reference):
    """
    Raise an Error if a required file is null or non-existent
    :param gff (str) path to alignment_summary.gff
    :param reference (str) path to reference_dir
    """
    if gff is None:
        raise PbReportError('gff cannot be None')
    if not os.path.exists(gff):
        raise IOError('gff {g} does not exist: '.format(g=gff))
    if reference is None:
        raise PbReportError('reference cannot be None')
    if not os.path.exists(reference):
        raise IOError('reference {g} does not exist: '.format(g=reference))


def _get_contigs_to_plot(alignment_summ_gff, contigs):
    """
    Returns a dict (string: ContigCoverage) that maps a contig ID to its coverage object.
    :param alignment_summ_gff: (str) path to alignment_summ_gff
    :param contigs: (list) top contigs from reference
    """

    def _get_name(id_):
        for c in contigs:
            if c.id == id_:
                return c.name

    cov_map = {}
    contig_ids = [c.id for c in contigs]

    reader = GffReader(alignment_summ_gff)
    for rec in reader:
        if rec.seqid not in contig_ids:
            log.info("Skipping seqid '{i}'.".format(i=rec.seqid))
            continue

        try:
            contig_cov = cov_map[rec.seqid]
        except KeyError:
            contig_cov = ContigCoverage(rec.seqid, _get_name(rec.seqid))
            cov_map[rec.seqid] = contig_cov

        contig_cov.add_data(rec)

    reader.close()

    return cov_map


def _create_contig_plot(contig_coverage):
    """
    Returns a fig,ax plot for this contig
    :param contig_coverage: (ContigCoverage) 
    """
    npXData = np.array(contig_coverage.xData)
    line_fill = LineFill(xData=npXData,
                         yData=np.array(contig_coverage.yDataMean),
                         linecolor=Constants.COLOR_STEEL_BLUE_DARK, alpha=0.6,
                         yDataMin=np.array(contig_coverage.yDataStdevMinus),
                         yDataMax=np.array(contig_coverage.yDataStdevPlus),
                         edgecolor=Constants.COLOR_STEEL_BLUE_LIGHT,
                         facecolor=Constants.COLOR_STEEL_BLUE_LIGHT)
    lines_fills = [line_fill]
    fig, ax = get_fig_axes_lpr()
    apply_line_data(ax, lines_fills, ('Reference Start Position', 'Coverage'))
    apply_line_fill_data(ax, lines_fills)
    return fig, ax


def _create_histogram(stats):
    """
    Returns a fig,ax histogram plot for this reference
    :param stats: (ReferenceStats) 
    """
    numBins = 100
    binSize = max(1, int(stats.maxbin / numBins))
    # handle case where the coverage is zero. This prevents the histogram
    # construction from crashing with an index error.
    m = 1 if stats.maxbin == 0.0 else stats.maxbin
    bins = np.arange(0, m, binSize)
    fig, ax = get_fig_axes_lpr()
    apply_histogram_data(ax, stats.means, bins,
                         ('Coverage', 'Reference Regions'),
                         barcolor=Constants.COLOR_STEEL_BLUE_DARK,
                         showEdges=False)
    return fig, ax


def _get_att_mean_coverage(stats):
    """
    :param stats (dict)
    """
    v = None
    if stats is None:
        v = 'NA'
    else:
        v = stats.mean_depth_of_coverage
    a = Attribute(Constants.A_COVERAGE, v,
                  name=Constants.ATTR_LABELS[Constants.A_COVERAGE])
    return a


def _get_att_percent_missing(stats):
    """
    :param stats (dict)
    """
    v = None
    if stats is None:
        v = 'NA'
    else:
        v = stats.perc_missing_bases
    a = Attribute(Constants.A_MISSING, v,
                  name=Constants.ATTR_LABELS[Constants.A_MISSING])
    return a


def _get_reference_coverage_stats(contigList):
    """Get a dictionary of coverage stats for the list of contigs"""

    maxbin = 0
    means = []
    cumulativeAveRegionSize = 0
    totalNumBases = 0
    totalMissingBases = 0

    contigCoverages = []
    numContigs = len(contigList)
    for cc in contigList:

        contigCoverages.append(cc.meanCoveragePerBase() * cc.numBases())
        cumulativeAveRegionSize += cc.aveRegionSize()

        if len(cc.yDataMean) > 0:
            means.extend(cc.yDataMean)
            localmax = max(cc.yDataMean)
            if localmax > maxbin:
                maxbin = localmax

        totalNumBases += cc.numBases()
        totalMissingBases += cc.missingBases()

    if totalNumBases == 0:
        log.warning(
            'totalNumBases is zero. Not able to calculate reference coverage stats.')
        return None

    mean_depth_of_coverage = sum(contigCoverages) / totalNumBases
    ave_region_size = int(cumulativeAveRegionSize / float(numContigs))
    perc_missing_bases = (float(totalMissingBases) /
                          float(totalNumBases)) * 100
    rs = ReferenceStats(maxbin, means,
                        mean_depth_of_coverage, ave_region_size,
                        perc_missing_bases)
    return rs


class ReferenceStats(object):

    def __init__(self, maxbin, means, mean_depth_of_coverage,
                 ave_region_size, perc_missing_bases):
        self._maxbin = maxbin
        self._means = means
        self._mean_depth_of_coverage = mean_depth_of_coverage
        self._ave_region_size = ave_region_size
        self._perc_missing_bases = perc_missing_bases

    @property
    def maxbin(self):
        return self._maxbin

    @property
    def means(self):
        return self._means

    @property
    def mean_depth_of_coverage(self):
        return self._mean_depth_of_coverage

    @property
    def ave_region_size(self):
        return self._ave_region_size

    @property
    def perc_missing_bases(self):
        return self._perc_missing_bases


class ContigCoverage(object):

    def __init__(self, seqid, name=None):
        """Encapsulates sequence info relevant to one chart"""

        self._seqid = seqid
        if name is None:
            name = seqid
        self._name = name

        self.xData = []

        self._numRecords = 0

        self._totalCoverage = 0
        self._refEnd = 0
        self._refStart = None

        self._cumulativeRegionSizes = 0
        self._numBases = 0
        self._missingBases = 0
        self._windowSize = 0

        self.yDataMean = []
        self.yDataStdevPlus = []
        self.yDataStdevMinus = []

        seqid_clean = re.sub("\|", "_", self._seqid) # for services
        self.file_name = "coverage_plot_{s}.png".format(s=seqid_clean)

    def __repr__(self):
        _d = dict(k=self.__class__.__name__, i=self._seqid, n=self.name,
                  s=self._refStart, e=self._refEnd, x=self._numRecords, b=self._numBases)
        return "<{k} {i} name:{n} ({s}, {e}) nrecords:{x} nbases:{b} >".format(**_d)

    def add_data(self, gff3Record):
        """Append x,y data from this record to the contig graph"""

        self._numRecords += 1

        if self._refStart is None:
            self._refStart = gff3Record.start

        self.xData.append(gff3Record.start)
        stats = gff3Record.attributes['cov2'].split(",")
        mean = float(stats[0])
        stddev = float(stats[1])

        self.yDataMean.append(mean)
        self.yDataStdevPlus.append(mean + stddev)

        # what is better - recalculating each time data is added, or storing 2
        # more arrays?
        regSize = (gff3Record.end - gff3Record.start) + 1

        self._totalCoverage += mean * regSize
        self._refEnd = gff3Record.end

        self._cumulativeRegionSizes = self._cumulativeRegionSizes + regSize

        # the second value of gaps pair is missing bases for region
        self._missingBases = self._missingBases + \
            int(gff3Record.attributes['gaps'].split(",")[1])

        # assumption: regions are continuous
        if self._numBases < gff3Record.end:
            self._numBases = gff3Record.end

        # clip at zero
        lowerBound = mean - stddev
        if lowerBound < 0:
            lowerBound = 0
        self.yDataStdevMinus.append(lowerBound)

    @property
    def name(self):
        return self._name

    def meanCoveragePerBase(self):
        """Get the normalized coverage per base"""
        if self._refStart is None:
            # contig wasn't found in gff
            return 0.0
        return self._totalCoverage / float(self._refEnd - self._refStart + 1)

    def aveRegionSize(self):
        """Get the average chunk size of this contig"""
        if self._numRecords == 0:
            return 0
        return int(self._cumulativeRegionSizes / float(self._numRecords))

    def missingBases(self):
        """Get number missing bases"""
        return self._missingBases

    def numBases(self):
        """Get number bases"""
        return self._numBases


def make_coverage_report(gff, reference, max_contigs_to_plot, report,
                         output_dir):
    """
    Entry to report.
    :param gff: (str) path to alignment_summary.gff
    :param reference: (str) path to reference_dir
    :param max_contigs_to_plot: (int) max number of contigs to plot
    """
    _validate_inputs(gff, reference)
    top_contigs = get_top_contigs(reference, max_contigs_to_plot)
    cov_map = _get_contigs_to_plot(gff, top_contigs)

    # stats may be None
    stats = _get_reference_coverage_stats(cov_map.values())

    a1 = _get_att_mean_coverage(stats)
    a2 = _get_att_percent_missing(stats)

    plot_grp_coverage = _create_coverage_plot_grp(
        top_contigs, cov_map, output_dir)

    plot_grp_histogram = None
    if stats is not None:
        plot_grp_histogram = _create_coverage_histo_plot_grp(stats, output_dir)

    plotgroups = []
    # Don't add the Plot Group if no plots are added
    if plot_grp_coverage.plots:
        plotgroups.append(plot_grp_coverage)

    if plot_grp_histogram is not None:
        # Don't add the Plot Group if no plots are added
        if plot_grp_histogram.plots:
            plotgroups.append(plot_grp_histogram)

    rpt = Report('coverage',
                 title="Coverage",
                 plotgroups=plotgroups,
                 attributes=[a1, a2],
                 dataset_uuids=(ReferenceSet(reference).uuid,))

    rpt.write_json(os.path.join(output_dir, report))
    return rpt


def args_runner(args):
    rpt = make_coverage_report(args.gff, args.reference, args.maxContigs,
                               args.report_json, op.dirname(args.report_json))
    log.info(rpt)
    return 0


def resolved_tool_contract_runner(rtc):
    rpt = make_coverage_report(
        gff=rtc.task.input_files[1],
        reference=rtc.task.input_files[0],
        max_contigs_to_plot=rtc.task.options[Constants.MAX_CONTIGS_ID],
        report=rtc.task.output_files[0],
        output_dir=op.dirname(rtc.task.output_files[0]))
    log.info(rpt)
    return 0


def get_parser():
    p = get_pbparser(
        tool_id=Constants.TOOL_ID,
        version=__version__,
        name="Coverage",
        description=__doc__,
        driver_exe=Constants.DRIVER_EXE,
        is_distributed=True)
    ap = p.arg_parser.parser
    p.add_input_file_type(FileTypes.DS_REF, "reference",
                          name="Reference DataSet",
                          description="Reference DataSet XML or FASTA file")
    p.add_input_file_type(FileTypes.GFF, "gff",
                          name="Alignment Summary GFF",
                          description="Alignment Summary GFF")
    p.add_output_file_type(FileTypes.REPORT, "report_json",
                           name="Coverage report",
                           description="Path to write report JSON output",
                           default_name="coverage_report")
    p.add_int(
        option_id=Constants.MAX_CONTIGS_ID,
        option_str="maxContigs",
        default=Constants.MAX_CONTIGS_DEFAULT,
        name="Maximum number of contigs to plot",
        description="Maximum number of contigs to plot in coverage report")
    return p


def main(argv=sys.argv[1:]):
    """Main point of Entry"""
    return pbparser_runner(argv,
                           get_parser(),
                           args_runner,
                           resolved_tool_contract_runner,
                           log,
                           setup_log)


if __name__ == '__main__':
    sys.exit(main())
