#!/usr/bin/env python3
import csv
import sys
import xml.etree.ElementTree as ET


def gather_info(testsuite):
    suite_name = testsuite.attrib['name']
    for testcase in testsuite:
        class_name = testcase.attrib['classname']
        name = testcase.attrib['name']
        time = testcase.attrib['time']
        yield [suite_name, class_name, name, time]


if __name__ == "__main__":
    with open(sys.argv[1], 'w') as output:
        writer = csv.writer(output, delimiter=',', quotechar='"')
        writer.writerow(["suite_name", "class_name", "name", "time"])
        for arg in sys.argv[2:]:
            parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
            report = ET.parse(arg, parser)
            for testsuite in report.getroot():
                for info in gather_info(testsuite):
                    writer.writerow(info)
