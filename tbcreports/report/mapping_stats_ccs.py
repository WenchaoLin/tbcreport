
"""
Generates a report of statistics for CCS reads mapped to a reference genome
with Blasr/pbalign.
"""

from collections import OrderedDict
import logging
import sys

from pbcommand.models import get_pbparser, FileTypes

from pbreports.report.streaming_utils import PlotViewProperties
from pbreports.plot.helper import (get_fig_axes_lpr, apply_line_data, Line)
from pbreports.report.mapping_stats import *
from pbreports.report.mapping_stats import Constants as BaseConstants
from pbreports.report.ccs import create_plot

class Constants(BaseConstants):
    TOOL_ID = "pbreports.tasks.mapping_stats_ccs"
    DRIVER_EXE = "python -m pbreports.report.mapping_stats_ccs --resolved-tool-contract"

__version__ = "0.1"
log = logging.getLogger(__name__)


def scatter_plot_accuracy_vs_concordance(
        data,
        axis_labels=("Predicted accuracy", "Mapped concordance"),
        nbins=None,
        barcolor=None):
    accuracy, concordance = data
    fig, ax = get_fig_axes_lpr()
    data = [Line(xData=accuracy,
                 yData=concordance,
                 style='+')]
    apply_line_data(
        ax=ax,
        line_models=data,
        axis_labels=axis_labels,
        only_whole_ticks=False)
    xlim = ax.get_xlim()
    xy = np.linspace(xlim[0], xlim[1])
    ax.plot(xy, xy, '-', color='r')
    return fig, ax


class CCSMappingStatsCollector(MappingStatsCollector):
    COLUMN_ATTR = [
        Constants.A_NREADS, Constants.A_READLENGTH, Constants.A_READLENGTH_N50,
        Constants.A_NBASES, Constants.A_READ_CONCORDANCE
    ]
    ATTR_LABELS = OrderedDict([
        (Constants.A_READ_CONCORDANCE, "Mapped CCS Read Mean Concordance"),
        (Constants.A_NREADS, "Number of CCS Reads (mapped)"),
        (Constants.A_NBASES, "Number of CCS Bases (mapped)"),
        (Constants.A_READLENGTH, "CCS Read Length Mean (mapped)"),
        (Constants.A_READLENGTH_N50, "CCS Read Length N50 (mapped)"),
        (Constants.A_READLENGTH_Q95, "CCS Read Length 95% (mapped)"),
        (Constants.A_READLENGTH_MAX, "CCS Read Length Max (mapped)"),
    ])
    ATTR_DESCRIPTIONS = {
        Constants.A_READ_CONCORDANCE: "The mean concordance of CCS reads that mapped to the reference sequence",
        Constants.A_NREADS: "The number of CCS reads that mapped to the reference sequence",
        Constants.A_NBASES: "The number of CCS read bases that mapped to the reference sequence",
        Constants.A_READLENGTH: "The mean length of CCS reads that mapped to the reference sequence",
        Constants.A_READLENGTH_Q95: "The 95th percentile of length of CCS reads that mapped to the reference sequence",
        Constants.A_READLENGTH_MAX: "The maximum length of CCS reads that mapped to the reference sequence",
        Constants.A_READLENGTH_N50: "The read length at which 50% of the bases are in reads longer than, or equal to, this value"
    }
    HISTOGRAM_IDS = {
        Constants.P_READ_CONCORDANCE: "read_concordance_histogram",
        Constants.P_READLENGTH: "readlength_histogram",
    }
    COLUMNS = [
        (Constants.C_MOVIE, "Movie"),
        (Constants.C_READS, "Number of CCS Reads (mapped)"),
        (Constants.C_READ_NBASES, "Number of CCS Bases (mapped)"),
        (Constants.C_READLENGTH, "CCS Read Length Mean (mapped)"),
        (Constants.C_READLENGTH_N50, "CCS Read Length N50 (mapped)"),
        (Constants.C_READ_CONCORDANCE, "Mapped CCS Read Mean Concordance")
    ]
    COLUMN_AGGREGATOR_CLASSES = [
        ReadCounterAggregator,
        MeanReadLengthAggregator,
        N50Aggreggator,
        NumberSubreadBasesAggregator,
        MeanSubreadConcordanceAggregator
    ]
    PG_QV_CALIBRATION = "qv_calibration"
    P_QV_CALIBRATION = "qv_calibration_plot"

    def _get_plot_view_configs(self):
        """
        There are just two histogram plots in this report:

        1. Mapped concordance
        2. Read length
        """
        _p = [
            PlotViewProperties(
                Constants.P_READ_CONCORDANCE,
                Constants.PG_READ_CONCORDANCE,
                generate_plot,
                'mapped_read_concordance_histogram.png',
                xlabel="Concordance",
                ylabel="CCS Reads",
                color=get_green(3),
                edgecolor=get_green(2),
                use_group_thumb=True,
                plot_group_title="Mapped CCS Read Concordance"),
            PlotViewProperties(
                Constants.P_READLENGTH,
                Constants.PG_READLENGTH,
                generate_plot,
                'mapped_readlength_histogram.png',
                xlabel="Read Length",
                ylabel="Reads",
                color=get_blue(3),
                edgecolor=get_blue(2),
                use_group_thumb=True,
                plot_group_title="Mapped CCS Read Length")
        ]
        return {v.plot_id: v for v in _p}

    def _get_total_aggregators(self):
        return OrderedDict([
            (Constants.A_READ_CONCORDANCE, MeanSubreadConcordanceAggregator()),
            (Constants.A_NREADS, ReadCounterAggregator()),
            (Constants.A_NBASES, NumberBasesAggregator()),
            (Constants.A_READLENGTH, MeanReadLengthAggregator()),
            (Constants.A_READLENGTH_N50, N50Aggreggator()),
            (Constants.A_READLENGTH_Q95, MappedReadLengthQ95(dx=10,
                                                             nbins=10000)),
            (Constants.A_READLENGTH_MAX, MaxReadLengthAggregator()),
            (self.HISTOGRAM_IDS[Constants.P_READLENGTH],
             ReadLengthHistogram()),
            (self.HISTOGRAM_IDS[Constants.P_READ_CONCORDANCE],
             SubReadConcordanceHistogram(dx=0.005, nbins=1001))
        ])

    def add_more_plots(self, plot_groups, output_dir):
        try:
            ds = ConsensusAlignmentSet(self.alignment_file)
            accuracy, concordance = [], []
            for bam in ds.resourceReaders():
                accuracy.extend(list(bam.pbi.readQual))
                concordance.extend(list(bam.identity))
            qv_validation_plot = create_plot(
                _make_plot_func=scatter_plot_accuracy_vs_concordance,
                plot_id=self.P_QV_CALIBRATION,
                axis_labels=("Predicted accuracy", "Mapped concordance"),
                nbins=None,
                plot_name="mapped_qv_calibration.png",
                barcolor="#A0A0FF",
                data=(accuracy, concordance),
                output_dir=output_dir)
            pg = PlotGroup(self.PG_QV_CALIBRATION,
                           title="Mapped QV Calibration",
                           plots=[qv_validation_plot],
                           thumbnail=qv_validation_plot.thumbnail)
            plot_groups.append(pg)
        except Exception as e:
            log.exception(e)
        return plot_groups


def to_report(alignment_file, output_dir):
    return CCSMappingStatsCollector(alignment_file).to_report(output_dir)


def _args_runner(args):
    return run_and_write_report(args.alignment_file, args.report_json,
                                report_func=to_report)


def _resolved_tool_contract_runner(resolved_contract):
    """
    Run the mapping report from a resolved tool contract.

    :param resolved_contract:
    :type resolved_contract: ResolvedToolContract
    :return: Exit code
    """

    alignment_path = resolved_contract.task.input_files[0]

    # This is a bit different than the output dir oddness that Johann added.
    # the reports dir should be determined from the dir of the abspath of the report.json file
    # resolved values will always be absolute paths.
    output_report = resolved_contract.task.output_files[0]

    return run_and_write_report(alignment_path, output_report,
                                report_func=to_report)


def _get_parser():
    parser = get_pbparser(Constants.TOOL_ID, __version__,
                          "CCS Mapping Statistics", __doc__,
                          Constants.DRIVER_EXE)
    parser.add_input_file_type(FileTypes.DS_ALIGN_CCS, "alignment_file",
                               "ConsensusAlignment XML DataSet",
                               "BAM, SAM or ConsensusAlignment DataSet")
    parser.add_output_file_type(FileTypes.REPORT, "report_json",
                                "PacBio Json Report",
                                "Output report JSON file.",
                                default_name="mapping_stats_report")
    return parser


if __name__ == '__main__':
    sys.exit(main(get_parser_func=_get_parser,
                  args_runner_func=_args_runner,
                  rtc_runner_func=_resolved_tool_contract_runner))
