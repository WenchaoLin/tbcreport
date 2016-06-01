#!/usr/bin/env python
"""Summarize the Long Amplicon Analysis using the ZMW results"""
import os
import sys
import logging
import argparse
from pprint import pformat
from collections import defaultdict

from pbcommand.models.report import Report, Table, Column
from pbcommand.models import FileTypes, get_pbparser
from pbcommand.cli import pbparser_runner
from pbcommand.common_options import add_debug_option
from pbcommand.utils import setup_log


log = logging.getLogger(__name__)

__version__ = '0.1.1'


class Constants(object):
    TOOL_ID = "pbreports.tasks.amplicon_analysis_consensus"

# TODO: I really shouldn't have this much logic in a report file.
#    this should be moved into FSharp as soon as reasonable


def parse_summary(summary):
    # Internal helper function for parsing the Summary CSV header
    def parse_summary_header(header):
        parts = header.strip().split(',')
        name = parts.index("FastaName")
        barcode = parts.index("BarcodeName")
        noise = parts.index("NoiseSequence")
        chimera = parts.index("IsChimera")
        return name, barcode, noise, chimera

    # Parse the summary file
    summary_data = {}
    with open(summary) as handle:
        # Read the summary header to find the location of important fields
        name, barcode, noise, chimera = parse_summary_header(
            handle.next().strip())

        for line in handle:
            # Parse the requisite fields from the current line
            parts = line.split(',')
            seq_name = parts[name]
            barcode_name = parts[barcode]
            noise_flag = True if parts[noise] == "True" else False
            chimera_flag = True if parts[chimera] == "True" else False

            # Catch whether it's a new barcode for summary setup
            if barcode_name not in summary_data:
                summary_data[barcode_name] = {
                    'chimera': set(), 'noise': set(), 'good': set()}

            # Add the current sequence to the appropriate bin
            if chimera_flag:
                summary_data[barcode_name]['chimera'].add(seq_name)
            elif noise_flag:
                summary_data[barcode_name]['noise'].add(seq_name)
            else:
                summary_data[barcode_name]['good'].add(seq_name)
    return summary_data


def parse_mappings(mappings_file):
    # Declare a helper function for parsing the heading of the ZMW mappings
    # file
    def parse_mappings_header(header):
        parts = header.strip().split(',')
        return {i: v for i, v in enumerate(parts[1:])}

    # Parse and sum the mapping weights from each ZMW
    position_sums = {}
    with open(mappings_file) as handle:
        consensus_positions = parse_mappings_header(handle.next())
        for line in handle:
            weights = [float(w) for w in line.strip().split(',')[1:]]
            for i, w in enumerate(weights):
                try:
                    position_sums[i] += w
                except:
                    position_sums[i] = w

    # Convert the sums-by-position into sums-by-consensus and return
    consensus_sums = {}
    for key, weight_sum in position_sums.iteritems():
        consensus = consensus_positions[key]
        consensus_sums[consensus] = weight_sum
    return consensus_sums


def tabulate_results(summary_data, consensus_sums):
    # Combine the individual consensus sums by barcode and in total
    tabulated_data = {'all': defaultdict(float)}
    for barcode, barcode_dict in summary_data.iteritems():
        data = defaultdict(float)
        for sequence_type, consensus_sequences in barcode_dict.iteritems():
            data[sequence_type] = 0.0
            for consensus in consensus_sequences:
                weight = consensus_sums[consensus]
                data[sequence_type] += weight
                tabulated_data['all'][sequence_type] += weight
        tabulated_data[barcode] = data

    # Round the final tallys and their percentages to the nearest integer
    final_data = {}
    for barcode, data in tabulated_data.iteritems():
        final_data[barcode] = defaultdict(float)
        total = data['good'] + data['chimeras'] + data['noise']
        for sequence_type, weight in data.iteritems():
            value = tabulated_data[barcode][sequence_type]
            final_data[barcode][sequence_type] = int(value)
            final_data[barcode][sequence_type + '_pct'] = value / total
    return final_data

# TODO: See note above this section


def create_table(tabulated_data):
    """Long Amplicon Analysis results table"""

    columns = []
    columns.append(Column("barcode_col", header="Sample"))
    columns.append(Column("good", header="Good"))
    columns.append(Column("good_pct", header="Good (%)"))
    columns.append(Column("chimera", header="Chimeric"))
    columns.append(Column("chimera_pct", header="Chimeric (%)"))
    columns.append(Column("noise", header="Noise"))
    columns.append(Column("noise_pct", header="Noise (%)"))

    t = Table("result_table",
              title="Amplicon Input Molecule Summary", columns=columns)

    for barcode, data in tabulated_data.iteritems():
        if barcode != 'all':
            t.add_data_by_column_id('barcode_col', barcode)
            for column_id in ['good', 'good_pct', 'chimera', 'chimera_pct', 'noise', 'noise_pct']:
                t.add_data_by_column_id(column_id, data[column_id])
    t.add_data_by_column_id('barcode_col', 'All')
    for column_id in ['good', 'good_pct', 'chimera', 'chimera_pct', 'noise', 'noise_pct']:
        t.add_data_by_column_id(column_id, tabulated_data['all'][column_id])

    log.info(str(t))
    return t


def run_to_report(summary_csv, zmw_csv):
    log.info("Generating PCR report v{v} from summary '{s}' "
             "and ZMW mappings '{m}'".format(v=__version__,
                                             s=summary_csv,
                                             m=zmw_csv))

    # Parse the data into dictionaries
    summary_data = parse_summary(summary_csv)
    consensus_sums = parse_mappings(zmw_csv)

    # Tabulate the values for each category
    tabulated_data = tabulate_results(summary_data, consensus_sums)

    # Convert the data into a PBreports table
    table = create_table(tabulated_data)

    # ids must be lowercase.
    r = Report("amplicon_analysis_input", tables=[table])

    return r


def amplicon_analysis_input(summary_csv, zmws_csv, report_json):
    log.info("Running {f} v{v}.".format(
        f=os.path.basename(__file__), v=__version__))
    report = run_to_report(summary_csv, zmws_csv)
    log.info(pformat(report.to_dict()))
    report.write_json(report_json)
    return 0


def args_runner(args):
    amplicon_analysis_input(args.report_csv, args.zmw_csv, args.report_json)
    return 0


def resolved_tool_contract_runner(resolved_tool_contract):
    rtc = resolved_tool_contract
    amplicon_analysis_input(rtc.task.input_files[0],
                            rtc.task.input_files[1],
                            rtc.task.output_files[0])
    return 0


def _add_options_to_parser(p):
    p.add_input_file_type(
        FileTypes.CSV,
        file_id="report_csv",
        name="ConsensusReportCSV",
        description="Consensus Report CSV")
    p.add_input_file_type(
        FileTypes.CSV,
        file_id="zmw_csv",
        name="ZMWReportCSV",
        description="ZMW Report CSV")
    p.add_output_file_type(
        FileTypes.JSON,
        file_id="report_json",
        name="ConsensusReportJSON",
        description="Consensus Report JSON",
        default_name="consensus_input_report")


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
    driver_exe = ("python -m "
                  "pbreports.report.amplicon_analysis_input "
                  "--resolved-tool-contract ")
    p = get_pbparser(
        Constants.TOOL_ID,
        __version__,
        "Amplicon Analysis Input",
        __doc__,
        driver_exe)
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

# for 'python -m pbreports.report.amplicon_analysis_input ...'
if __name__ == "__main__":
    sys.exit(main())
