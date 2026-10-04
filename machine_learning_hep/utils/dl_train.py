#!/bin/env python3

"""This module downloads AO2Ds from an ALICE hyperloop train run"""

import argparse
import os
from pathlib import PurePosixPath
from time import localtime, strftime

import requests  # pylint: disable=import-error

try:
    from alienpy import alien, xrd_core
except ImportError:
    print("Could not import alien, install with pip install alienpy")


def get_train_spec(train_id: int):
    """Retrieve train spec from hyperloop interface"""
    # https://alimonitor.cern.ch/hyperloop/train-run/131050
    url = f"https://alimonitor.cern.ch/alihyperloop-data/trains/train.jsp?train_id={train_id}"
    try:
        return requests.get(
            url,
            verify=False,
            cert=(f"/tmp/tokencert_{os.getuid()}.pem", f"/tmp/tokenkey_{os.getuid()}.pem"),
            timeout=10,
        )
    except requests.exceptions.SSLError as e:
        print(f"SSL Error: {e}")
        raise


def find_ao2ds(ali: alien.AliEn, aliendir: str) -> list[str]:
    """Find AO2Ds in train output directory"""
    cmd_find = f"find {PurePosixPath(aliendir) / 'AOD'} AO2D.root"
    print(cmd_find)
    ret = ali.run(cmd_find)
    if ret.exitcode != 0:
        print(f"Failed to run search: {cmd_find}\n{ret.out}")
        cmd_find = f"find {PurePosixPath(aliendir)} AO2D.root"
        print(cmd_find)
        ret = ali.run(cmd_find)
        if ret.exitcode != 0:
            print(f"Failed to run search: {cmd_find}\n{ret.out}")
            return []
    return ret.out.split()


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(description="Download AO2Ds from hyperloop train")
    parser.add_argument("train_id", type=int, nargs="+", help="train IDs")
    parser.add_argument("--prefix", "-p", default="/data2/MLhep/trains/", help="destination directory")
    parser.add_argument("--print-specs", "-s", action="store_true", help="print train specs and exit")
    parser.add_argument("--dry-run", "-n", action="store_true", help="dry run")
    args = parser.parse_args()

    if args.print_specs:
        print("Specs:\tID\tsubmitted\tO2Physics\tdataset")

    for train_id in args.train_id:
        print(f"Obtaining train spec for train {train_id}...")
        train_spec = get_train_spec(train_id).json()
        if args.print_specs:
            time_submitted = strftime("%Y-%m-%d", localtime(train_spec["train_submitted"] / 1000))
            dataset = train_spec["dataset_name"]
            package_tag = train_spec["package_tag"][26:34]
            package_tag = f"{package_tag[:4]}-{package_tag[4:6]}-{package_tag[6:]}"
            # output_size = train_spec["output_size"] # TODO: How to interpret the value?
            print(f"Specs:\t{train_id}\t{time_submitted}\t{package_tag}\t{dataset}")
            continue
        print("Finding AO2Ds...")
        outputdirs = [d["outputdir"] for d in train_spec["jobResults"]]
        a = alien.AliEn()
        src = [d for outputdir in outputdirs for d in find_ao2ds(a, outputdir)]
        dst = ["file:" + str(PurePosixPath(args.prefix) / str(train_id) / file.lstrip("/")) for file in src]
        print("Files to copy:")
        for s, d in zip(src, dst, strict=False):
            print(f"{s} -> {d}")

        if not args.dry_run:
            print("Copying...")
            xrd_core.DO_XrootdCp(a.wb(), api_src=src, api_dst=dst)


if __name__ == "__main__":
    main()
