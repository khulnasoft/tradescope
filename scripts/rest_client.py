#!/usr/bin/env python3
"""
Simple command line client into RPC commands
Can be used as an alternate to Telegram

Should not import anything from tradescope,
so it can be used as a standalone script.
"""

from tradescope_client.ts_client import main


if __name__ == '__main__':
    main()
