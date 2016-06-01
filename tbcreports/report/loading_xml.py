
"""
Generate XML report on ZMW loading and productivity.
"""

import os
import logging
import sys
import numpy as np

from pbcommand.models.report import Report, Table, Column
from pbcommand.models import TaskTypes, FileTypes, get_pbparser
from pbcommand.cli import pbparser_runner
from pbcommand.common_options import add_debug_option
from pbcommand.utils import setup_log
from pbcore.io import DataSet

__version__ = '0.1.0'


class Constants(object):
    TOOL_ID = "pbreports.tasks.loading_report_xml"
    DRIVER_EXE = ("python -m pbreports.report.loading_xml "
                  "--resolved-tool-contract ")
    DECIMALS = 3


log = logging.getLogger(__name__)


def to_report(stats_xml):
    """Main point of entry

    :type stats_xml: str
    :type output_dir: str
    :type dpi: int

    :rtype: Report
    """
    log.info("Analyzing XML {f}".format(f=stats_xml))
    dset = DataSet(stats_xml)
    if not dset.metadata.summaryStats:
        dset.loadStats(stats_xml)
    if not dset.metadata.summaryStats.prodDist:
        raise IOError("Pipeline Summary Stats (sts.xml) not found or missing "
                      "key distributions")

    dsets = [dset]
    for subdset in dset.subdatasets:
        if subdset.metadata.summaryStats:
            dsets.append(subdset)

    col_names = ["Collection Context",
                 "Productive ZMWs",
                 "Productivity 0 (%)",
                 "Productivity 1 (%)",
                 "Productivity 2 (%)"]
    col_values = [[], [], [], [], []]
    for dset in dsets:
        if len(dsets) > 1 and len(col_values[0]) == 0:
            movie_name = "Combined"
        else:
            try:
                collection = list(dset.metadata.collections)[0]
                movie_name = collection.context
            except AttributeError:
                movie_name = "NA"

        productive_zmws = int(dset.metadata.summaryStats.numSequencingZmws)
        empty, productive, other, _ = dset.metadata.summaryStats.prodDist.bins

        prod0 = np.round(100.0 * empty / float(productive_zmws),
                         decimals=Constants.DECIMALS)
        prod1 = np.round(100.0 * productive / float(productive_zmws),
                         decimals=Constants.DECIMALS)
        prod2 = np.round(100.0 * other / float(productive_zmws),
                         decimals=Constants.DECIMALS)
        this_row = [movie_name, productive_zmws, prod0, prod1, prod2]
        map(lambda (x, y): x.append(y), zip(col_values, this_row))
    columns = [Column(cn.translate(None, '(%)').strip().replace(' ',
                                                                '_').lower(),
                      cn, vals)
               for cn, vals in zip(col_names, col_values)]
    tables = [Table("loading_xml_table", "Loading Statistics", columns)]
    report = Report("loading_xml_report", title="Loading Report",
                    tables=tables, attributes=None, plotgroups=None)
    return report


def args_runner(args):
    log.info("Starting {f} v{v}".format(f=os.path.basename(__file__),
                                        v=__version__))
    report = to_report(args.subread_set)
    report.write_json(args.report)
    return 0


def resolved_tool_contract_runner(resolved_tool_contract):
    rtc = resolved_tool_contract
    log.info("Starting {f} v{v}".format(f=os.path.basename(__file__),
                                        v=__version__))
    report = to_report(rtc.task.input_files[0])
    report.write_json(rtc.task.output_files[0])
    return 0


def _add_options_to_parser(p):
    p.add_input_file_type(
        FileTypes.DS_SUBREADS,
        file_id="subread_set",
        name="SubreadSet",
        description="SubreadSet")
    p.add_output_file_type(FileTypes.REPORT, "report", "Loading report",
                           description=("Filename of JSON output report. Should be name only, "
                                        "and will be written to output dir"),
                           default_name="report")


def add_options_to_parser(p):
    """
    API function for extending main pbreport arg parser (independently of
    tool contract interface).
    """
    p_wrap = _get_parser_core()
    p_wrap.arg_parser.parser = p
    p.description = __doc__
    add_debug_option(p)
    _add_options_to_parser(p_wrap)
    p.set_defaults(func=args_runner)
    return p


def _get_parser_core():
    p = get_pbparser(
        Constants.TOOL_ID,
        __version__,
        "Loading XML Report",
        __doc__,
        Constants.DRIVER_EXE,
        is_distributed=True)
    return p


def get_parser():
    p = _get_parser_core()
    _add_options_to_parser(p)
    return p


def main(argv=sys.argv):
    mp = get_parser()
    return pbparser_runner(argv[1:],
                           mp,
                           args_runner,
                           resolved_tool_contract_runner,
                           log,
                           setup_log)


# for 'python -m pbreports.report.sat ...'
if __name__ == "__main__":
    sys.exit(main())
