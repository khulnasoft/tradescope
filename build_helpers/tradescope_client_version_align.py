#!/usr/bin/env python3
from tradescope_client import __version__ as client_version

from tradescope import __version__ as ts_version


def main():
    if ts_version != client_version:
        print(f"Versions do not match: \n"
              f"ft: {ts_version} \n"
              f"client: {client_version}")
        exit(1)
    print(f"Versions match: ft: {ts_version}, client: {client_version}")
    exit(0)


if __name__ == '__main__':
    main()
